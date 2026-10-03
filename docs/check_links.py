#!/usr/bin/env python3
"""Check links in the documentation sources and fail on broken ones.

The checker is split into two tiers so the per-PR gate is deterministic and the
flaky part (reaching the public internet) runs as a scheduled monitor instead:

* **Internal links** -- relative paths to other docs, images, and downloadable
  assets (e.g. ``example_configs/ni_bulk.xyz``). The target must exist on disk
  relative to the linking file. This is deterministic, so a broken one is a
  hard ERROR. The per-PR docs build runs ``--no-external`` (see
  ``.github/workflows/build_docs.yml``) so a new dead relative link fails CI
  immediately and offline.
* **External links** -- ``http(s)`` URLs are fetched concurrently. A definitively
  dead link (HTTP 404/410 or a DNS failure) is a hard ERROR; transient or
  access-gated responses (401/403/429/5xx, timeouts) are reported as WARN /
  "unverified" so the monitor is not flaky on rate-limiting or login-gated pages
  (e.g. HuggingFace gated models). These run on a schedule
  (``.github/workflows/check_links_external.yml``), not on every PR.

Link syntaxes understood (over all ``docs/**/*.md`` sources, ignoring fenced
code and ``{code-cell}`` blocks so example URLs inside code are not flagged):

* inline links / images ``[text](target)`` / ``![alt](target)`` -- with
  balanced-parenthesis-aware target capture (so URLs like
  ``.../Article_(disambiguation)`` are not truncated);
* autolinks ``<https://...>``;
* reference-style definitions ``[label]: target``;
* MyST ``:link:`` directive options (used by ``{grid-item-card}`` / ``{card}``);
* raw HTML ``<a href="...">`` and ``<img src="...">``.

Exit code is non-zero iff there is at least one ERROR.

Usage:
    python docs/check_links.py                 # internal + external
    python docs/check_links.py --no-external    # internal only (CI PR gate, offline)
    python docs/check_links.py --workers 16     # concurrency for external fetches
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urlsplit
from urllib.request import Request, urlopen

# --- link extraction -------------------------------------------------------

# autolinks: <https://...>
_AUTOLINK_RE = re.compile(r"<((?:https?)://[^>\s]+)>")
# fenced code blocks ``` ... ``` or ~~~ ... ~~~ (incl. ```{code-cell} ...```).
# NOTE: MyST ``:::`` colon-fence directives (grid-item-card, admonitions) are
# intentionally NOT stripped -- their ``:link:`` options are real links.
_FENCE_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})")
# inline code spans `...`
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
# reference-style link definition at line start: [label]: target
_REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)")
# MyST directive option: ":link: target"
_MYST_LINK_RE = re.compile(r"^\s*:link:\s*(\S+)")
# raw HTML href/src attributes
_HTML_ATTR_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

# External responses that must NOT fail the build (gated / transient / flaky).
_TOLERATED_STATUS = {401, 403, 429, 500, 502, 503, 504, 408}


def strip_code(text: str) -> str:
    """Remove fenced code blocks and inline code spans so their URLs are ignored."""
    out, fence = [], None
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if fence is None and m:
            fence = m.group(2)[0]
            continue
        if fence is not None:
            if line.strip().startswith(fence * 3):
                fence = None
            continue
        out.append(line)
    joined = "\n".join(out)
    return _INLINE_CODE_RE.sub("", joined)


def _inline_targets(body: str) -> list[str]:
    """Extract targets from ``[text](target)`` / ``![alt](target)``.

    Scans for the ``](`` opener and matches balanced parentheses so that URLs
    containing balanced ``()`` (e.g. Wikipedia disambiguation links) are kept
    whole. The target ends at the first whitespace (the optional "title").
    """
    out: list[str] = []
    for m in re.finditer(r"!?\[[^\]]*\]\(", body):
        j, depth, buf = m.end(), 1, []
        while j < len(body) and depth > 0:
            c = body[j]
            if c == "(":
                depth += 1
                buf.append(c)
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
                buf.append(c)
            elif c.isspace():
                break  # start of the optional title -> target is complete
            else:
                buf.append(c)
            j += 1
        tgt = "".join(buf).strip()
        if tgt.startswith("<") and tgt.endswith(">"):
            tgt = tgt[1:-1].strip()
        if tgt:
            out.append(tgt)
    return out


def _looks_like_path(target: str) -> bool:
    """Heuristic: is a reference-style target an actual link (not prose)?

    Reference-definition lines can occasionally capture non-link text; only
    treat a target as a checkable internal path if it is clearly path-like, to
    avoid false ERRORs in the blocking gate.
    """
    if target.startswith(("http://", "https://", "/", "./", "../", "#")):
        return True
    return "/" in target or "." in target


def extract_links(text: str) -> list[str]:
    """Return every link target found in ``text`` (code stripped)."""
    body = strip_code(text)
    links: list[str] = []
    links.extend(_inline_targets(body))
    links.extend(_AUTOLINK_RE.findall(body))
    for line in body.splitlines():
        if m := _MYST_LINK_RE.match(line):
            links.append(m.group(1).strip())
        if m := _REF_DEF_RE.match(line):
            tgt = m.group(1).strip()
            if _looks_like_path(tgt):
                links.append(tgt)
    links.extend(_HTML_ATTR_RE.findall(body))
    return links


def is_checkable_internal(target: str) -> bool:
    if target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
        return False
    # template placeholders / role syntax we cannot resolve on disk
    if target.startswith(("{", "%")) or "{{" in target or " " in target:
        return False
    return True


# --- checks ----------------------------------------------------------------


def check_internal(docs_dir: Path, src: Path, target: str) -> str | None:
    """Return an error string if the internal target does not exist, else None."""
    path_part = urldefrag(target)[0]
    path_part = path_part.split("?")[0]
    if not path_part:
        return None  # pure anchor
    if path_part.startswith("/"):
        candidate = (docs_dir / path_part.lstrip("/")).resolve()
    else:
        candidate = (src.parent / path_part).resolve()
    if candidate.exists():
        return None
    # allow extension-less links that map to a .md or .ipynb page (MyST pages)
    if (candidate.parent / (candidate.name + ".md")).exists():
        return None
    if (candidate.parent / (candidate.name + ".ipynb")).exists():
        return None
    return f"{src.relative_to(docs_dir)} -> {target} (no file at {candidate})"


def check_external(url: str, timeout: int, retries: int) -> tuple[str, str]:
    """Return (level, message). level in {'ok','warn','error'}."""
    base = urldefrag(url)[0]
    if not urlsplit(base).netloc:
        return "warn", f"{url} (unparseable)"
    last = ""
    for attempt in range(retries + 1):
        # HEAD is cheap but many servers mis-handle it (404/405 on HEAD while GET
        # is 200). Only a GET 404/410 is treated as authoritatively dead.
        for method in ("HEAD", "GET"):
            try:
                req = Request(
                    base,
                    method=method,
                    headers={"User-Agent": "Mozilla/5.0 (fairchem-docs-linkcheck)"},
                )
                with urlopen(req, timeout=timeout) as resp:
                    code = resp.getcode()
                if code and code < 400:
                    return "ok", f"{url} [{code}]"
                last = f"{url} [{code}]"
            except HTTPError as e:
                if e.code < 400:
                    return "ok", f"{url} [{e.code}]"
                if e.code in (404, 410):
                    if method == "GET":  # authoritative -> dead
                        return "error", f"{url} [{e.code}]"
                    last = f"{url} [{e.code} on HEAD]"  # let GET decide
                    continue
                if e.code in _TOLERATED_STATUS:
                    last = f"{url} [{e.code} tolerated]"
                    continue
                last = f"{url} [{e.code}]"
            except (URLError, TimeoutError, OSError) as e:
                last = f"{url} ({e})"
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    # No clear 2xx/3xx and no authoritative GET 404. A hard DNS/connection
    # failure is fatal; anything else (timeouts, HEAD-only 404, gated) is a warn.
    low = last.lower()
    if "name or service not known" in low or "nodename nor servname" in low or "no address" in low:
        return "error", f"{last} (DNS failure)"
    return "warn", f"{last} (unverified)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--external", dest="external", action="store_true", default=True)
    ap.add_argument("--no-external", dest="external", action="store_false")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16, help="concurrent external fetches")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    md_files = sorted(docs_dir.rglob("*.md"))
    print(f"Scanning {len(md_files)} markdown files under {docs_dir}")

    errors: list[str] = []
    warnings: list[str] = []
    external: dict[str, list[Path]] = {}
    internal_checked = 0

    for src in md_files:
        if "_build" in src.parts:
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for target in extract_links(text):
            if target.startswith(("http://", "https://")):
                external.setdefault(target, []).append(src)
            elif is_checkable_internal(target):
                internal_checked += 1
                err = check_internal(docs_dir, src, target)
                if err:
                    errors.append(f"[internal] {err}")

    internal_broken = len(errors)
    print(
        f"Internal: {internal_checked} link(s) checked, {internal_broken} broken. "
        f"{len(external)} unique external URL(s) found."
    )

    ext_ok = ext_warn = ext_err = 0
    if args.external:
        def _probe(item: tuple[str, list[Path]]) -> tuple[str, str, str]:
            url, srcs = item
            level, msg = check_external(url, args.timeout, args.retries)
            where = ", ".join(sorted({str(s.relative_to(docs_dir)) for s in srcs}))
            return level, msg, where

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for level, msg, where in pool.map(_probe, sorted(external.items())):
                if level == "error":
                    ext_err += 1
                    errors.append(f"[external] {msg}  (in {where})")
                elif level == "warn":
                    ext_warn += 1
                    warnings.append(f"[external] {msg}  (in {where})")
                else:
                    ext_ok += 1
        print(
            f"External: {ext_ok} verified, {ext_warn} unverified "
            f"(gated/transient), {ext_err} broken (of {len(external)})."
        )
    else:
        print("Skipping external link checks (--no-external).")

    if warnings:
        print(f"\n{len(warnings)} warning(s) (not failing the build):")
        for w in warnings:
            print(f"  WARN  {w}")

    if errors:
        print(f"\n{len(errors)} broken link(s):")
        for e in errors:
            print(f"  ERROR {e}")
        print("\nFAILED: broken links found.")
        return 1

    # Deliberately NOT claiming "all links OK" when some were only unverified.
    if ext_warn:
        print(
            f"\nNo broken links. ({ext_warn} external link(s) unverified -- "
            "gated/transient, not confirmed.)"
        )
    else:
        print("\nAll links OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
