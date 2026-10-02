#!/usr/bin/env python3
"""Check every link in the documentation sources and fail on broken ones.

Run as part of the docs build (see .github/workflows/build_docs.yml) so a PR that
introduces a dead link fails CI instead of silently shipping a 404.

What is checked (over all ``docs/**/*.md`` sources, ignoring fenced code and
``{code-cell}`` blocks so example URLs inside code are not flagged):

* Internal links -- relative paths to other docs, images, and downloadable
  assets (e.g. ``example_configs/ni_bulk.xyz``). The target must exist on disk
  relative to the linking file. These are deterministic -> a broken one is a
  hard ERROR (fatal).
* External links -- ``http(s)`` URLs are fetched. A definitively dead link
  (HTTP 404/410 or a DNS/connection failure) is a hard ERROR. Transient or
  access-gated responses (401/403/429/5xx, timeouts) are reported as WARN only,
  so CI is not flaky on rate-limiting or login-gated pages (e.g. HuggingFace
  gated models).

Exit code is non-zero iff there is at least one ERROR.

Usage:
    python docs/check_links.py                 # internal + external (CI default)
    python docs/check_links.py --no-external   # internal links only (offline)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urlsplit
from urllib.request import Request, urlopen

# --- link extraction -------------------------------------------------------

# inline markdown links and images: [text](target)  /  ![alt](target)
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)")
# autolinks: <https://...>
_AUTOLINK_RE = re.compile(r"<((?:https?)://[^>\s]+)>")
# fenced code blocks ``` ... ``` or ~~~ ... ~~~ (incl. ```{code-cell} ...```)
_FENCE_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})")
# inline code spans `...`
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

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


def extract_links(text: str) -> list[str]:
    body = strip_code(text)
    links = []
    for m in _LINK_RE.finditer(body):
        tgt = m.group(1).strip()
        if tgt.startswith("<") and tgt.endswith(">"):
            tgt = tgt[1:-1].strip()
        links.append(tgt)
    links.extend(_AUTOLINK_RE.findall(body))
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
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    md_files = sorted(docs_dir.rglob("*.md"))
    print(f"Scanning {len(md_files)} markdown files under {docs_dir}")

    errors: list[str] = []
    warnings: list[str] = []
    external: dict[str, list[Path]] = {}

    for src in md_files:
        if "_build" in src.parts:
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for target in extract_links(text):
            if target.startswith(("http://", "https://")):
                external.setdefault(target, []).append(src)
            elif is_checkable_internal(target):
                err = check_internal(docs_dir, src, target)
                if err:
                    errors.append(f"[internal] {err}")

    print(f"Internal links checked. {len(external)} unique external URLs found.")

    if args.external:
        for i, (url, srcs) in enumerate(sorted(external.items()), 1):
            level, msg = check_external(url, args.timeout, args.retries)
            where = ", ".join(sorted({str(s.relative_to(docs_dir)) for s in srcs}))
            if level == "error":
                errors.append(f"[external] {msg}  (in {where})")
            elif level == "warn":
                warnings.append(f"[external] {msg}  (in {where})")
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

    print("\nAll links OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
