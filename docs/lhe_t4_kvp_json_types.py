#!/usr/bin/env python3
"""Recursive aggregate-only shape audit for JSON in local Hyper-V KVP pools."""

from __future__ import annotations

import collections
import ipaddress
import json
import math
import re
import urllib.parse
from pathlib import Path
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_KVP_JSON"
ROOT = Path("/var/lib/hyperv")
KEY_SIZE = 512
VALUE_SIZE = 2048
RECORD_SIZE = KEY_SIZE + VALUE_SIZE
MAX_RECORDS = 4096
MAX_NODES = 200_000
MAX_DEPTH = 20
GUID_RE = re.compile(
    r"^\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?$"
)
JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")
BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = collections.Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def key_class(key: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", key.casefold())
    if any(
        token in compact
        for token in (
            "token",
            "secret",
            "password",
            "credential",
            "privatekey",
            "accountkey",
            "sharedkey",
            "sasuri",
            "signature",
        )
    ):
        return "sensitive_name"
    if "protectedsettings" in compact or "encryptedsettings" in compact:
        return "protected_settings"
    if "certificate" in compact or "thumbprint" in compact:
        return "certificate_metadata"
    if any(token in compact for token in ("url", "uri", "endpoint")):
        return "endpoint"
    if any(token in compact for token in ("identity", "vmid", "machineid", "subscriptionid", "tenantid")):
        return "identity"
    if "extension" in compact or "provision" in compact:
        return "extension_or_provisioning"
    if "status" in compact or "health" in compact or "telemetry" in compact:
        return "status_or_telemetry"
    if "host" in compact or "fabric" in compact or "azure" in compact:
        return "host_or_fabric"
    if "network" in compact or "address" in compact or "gateway" in compact:
        return "network"
    return "other"


def string_shape(value: str) -> tuple[str, bool, str | None]:
    stripped = value.strip()
    lowered = stripped.casefold()
    if not stripped:
        return "empty", False, None
    if GUID_RE.fullmatch(stripped):
        return "guid", False, None
    if JWT_RE.fullmatch(stripped):
        return "jwt", True, None
    if stripped.startswith("-----BEGIN "):
        return "pem", True, None
    if lowered.startswith(("bearer ", "basic ")):
        return "authorization_header", True, None
    try:
        address = ipaddress.ip_address(stripped.strip("[]"))
        if address.is_link_local:
            return "ip_link_local", False, None
        if address.is_private:
            return "ip_private", False, None
        if address.is_global:
            return "ip_global", False, None
        return "ip_other", False, None
    except ValueError:
        pass
    try:
        parsed = urllib.parse.urlsplit(stripped)
    except ValueError:
        parsed = urllib.parse.SplitResult("", "", "", "", "")
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = parsed.hostname.casefold().rstrip(".")
        if host == "168.63.129.16":
            host_class = "azure_wire_server"
        elif host.endswith(".blob.core.windows.net"):
            host_class = "azure_blob"
        elif host.endswith((".azure.com", ".azure.net", ".windows.net")):
            host_class = "azure"
        elif host.endswith((".github.com", ".github.net")):
            host_class = "github"
        else:
            host_class = "other"
        query_names = {name.casefold() for name, _ in urllib.parse.parse_qsl(parsed.query)}
        signed = bool(query_names & {"sig", "signature", "token", "key", "secret", "se", "sp"})
        return "signed_url" if signed else "url", signed, host_class
    if len(stripped) >= 40 and BASE64ISH_RE.fullmatch(stripped) and entropy(stripped) >= 4.2:
        return "opaque_high_entropy", False, None
    if len(stripped) >= 40 and HEX_RE.fullmatch(stripped) and entropy(stripped) >= 3.2:
        return "opaque_hex", False, None
    credential_marker = bool(
        re.search(
            r"(?i)(?:token|secret|password|signature|credential|private[_-]?key)"
            r"[\"']?\s*[:=]",
            stripped,
        )
    )
    signed_url_marker = bool(
        re.search(r"(?i)https?://[^\s]+[?&](?:sig|token|key|secret|se|sp)=", stripped)
    )
    if credential_marker or signed_url_marker:
        return "embedded_credential_marker", True, None
    if stripped.startswith(("/", "./", "../")) or re.match(r"^[A-Za-z]:[\\/]", stripped):
        return "path", False, None
    return "text", False, None


def audit_json(root: Any) -> dict[str, Any]:
    key_counts: collections.Counter[str] = collections.Counter()
    shape_counts: collections.Counter[str] = collections.Counter()
    sensitive_shape_counts: collections.Counter[str] = collections.Counter()
    host_counts: collections.Counter[str] = collections.Counter()
    credential_shapes = 0
    sensitive_nonempty_scalars = 0
    nodes = 0
    max_depth_seen = 0
    pending: list[tuple[Any, int, str | None]] = [(root, 0, None)]
    while pending and nodes < MAX_NODES:
        value, depth, parent_key_class = pending.pop()
        nodes += 1
        max_depth_seen = max(max_depth_seen, depth)
        if depth > MAX_DEPTH:
            continue
        if isinstance(value, dict):
            for key, child in value.items():
                classification = key_class(str(key))
                key_counts[classification] += 1
                pending.append((child, depth + 1, classification))
        elif isinstance(value, list):
            pending.extend((child, depth + 1, parent_key_class) for child in value)
        elif isinstance(value, str):
            shape, credential, host_class = string_shape(value)
            shape_counts[shape] += 1
            credential_shapes += int(credential)
            if host_class:
                host_counts[host_class] += 1
            if parent_key_class in {"sensitive_name", "protected_settings"}:
                sensitive_shape_counts[shape] += 1
                sensitive_nonempty_scalars += int(bool(value))
        elif value is None:
            shape_counts["null"] += 1
        elif isinstance(value, bool):
            shape_counts["boolean"] += 1
        elif isinstance(value, (int, float)):
            shape_counts["number"] += 1
        else:
            shape_counts["other_scalar"] += 1
    return {
        "node_count": nodes,
        "node_limit_reached": bool(pending),
        "max_depth_seen": max_depth_seen,
        "nested_key_class_counts": dict(sorted(key_counts.items())),
        "nested_string_shape_counts": dict(sorted(shape_counts.items())),
        "sensitive_key_value_shape_counts": dict(sorted(sensitive_shape_counts.items())),
        "sensitive_nonempty_scalar_count": sensitive_nonempty_scalars,
        "credential_shape_count": credential_shapes,
        "url_host_class_counts": dict(sorted(host_counts.items())),
    }


def inspect_pool(index: int) -> dict[str, Any]:
    path = ROOT / f".kvp_pool_{index}"
    output: dict[str, Any] = {
        "present": path.is_file(),
        "record_count": 0,
        "json_record_count": 0,
        "json_parse_failure_count": 0,
    }
    if not path.is_file():
        return output
    try:
        size = path.stat().st_size
        if size % RECORD_SIZE or size > RECORD_SIZE * MAX_RECORDS:
            output["read_refused"] = True
            return output
        data = path.read_bytes()
    except OSError as exc:
        output["read_error_class"] = type(exc).__name__
        return output

    aggregates: collections.Counter[str] = collections.Counter()
    key_counts: collections.Counter[str] = collections.Counter()
    shape_counts: collections.Counter[str] = collections.Counter()
    sensitive_shapes: collections.Counter[str] = collections.Counter()
    host_counts: collections.Counter[str] = collections.Counter()
    for offset in range(0, len(data), RECORD_SIZE):
        record = data[offset : offset + RECORD_SIZE]
        key = c_string(record[:KEY_SIZE])
        if not key:
            continue
        output["record_count"] += 1
        value = c_string(record[KEY_SIZE:])
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            output["json_parse_failure_count"] += 1
            continue
        output["json_record_count"] += 1
        audited = audit_json(parsed)
        aggregates["node_count"] += audited["node_count"]
        aggregates["credential_shape_count"] += audited["credential_shape_count"]
        aggregates["sensitive_nonempty_scalar_count"] += audited[
            "sensitive_nonempty_scalar_count"
        ]
        aggregates["node_limit_record_count"] += int(audited["node_limit_reached"])
        aggregates["max_depth_seen"] = max(
            aggregates["max_depth_seen"], audited["max_depth_seen"]
        )
        key_counts.update(audited["nested_key_class_counts"])
        shape_counts.update(audited["nested_string_shape_counts"])
        sensitive_shapes.update(audited["sensitive_key_value_shape_counts"])
        host_counts.update(audited["url_host_class_counts"])
    output.update(dict(aggregates))
    output["nested_key_class_counts"] = dict(sorted(key_counts.items()))
    output["nested_value_shape_counts"] = dict(sorted(shape_counts.items()))
    output["sensitive_key_value_shape_counts"] = dict(sorted(sensitive_shapes.items()))
    output["url_host_class_counts"] = dict(sorted(host_counts.items()))
    return output


def main() -> int:
    pools = {str(index): inspect_pool(index) for index in (1, 3)}
    result = {
        "schema": "meta-lhe-fairchem-kvp-json-types-v1",
        "scope": "recursive aggregate-only JSON shape audit for local KVP pools",
        "pools": pools,
        "credential_shape_total": sum(
            int(pool.get("credential_shape_count", 0)) for pool in pools.values()
        ),
        "raw_key_emitted": False,
        "raw_value_emitted": False,
        "json_body_emitted": False,
        "hostname_address_url_or_identifier_emitted": False,
        "network_request_sent": False,
        "vmbus_request_sent": False,
        "external_state_changed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
