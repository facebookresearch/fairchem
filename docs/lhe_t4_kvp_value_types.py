#!/usr/bin/env python3
"""Aggregate-only classifier for local Hyper-V KVP record value shapes."""

from __future__ import annotations

import collections
import ipaddress
import json
import math
import re
import urllib.parse
from pathlib import Path
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_KVP_TYPES"
ROOT = Path("/var/lib/hyperv")
KEY_SIZE = 512
VALUE_SIZE = 2048
RECORD_SIZE = KEY_SIZE + VALUE_SIZE
MAX_RECORDS = 4096
GUID_RE = re.compile(
    r"^\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?$"
)
JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")
VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,7}$")
INTEGER_RE = re.compile(r"^-?\d+$")
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?$"
)
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
    lowered = key.casefold()
    if lowered in {"hostname", "physicalhostname", "physicalhostnamefullyqualified"}:
        return "standard_host_name"
    if lowered.startswith("hostingsystem"):
        return "standard_host_system_property"
    if lowered.startswith("virtualmachine"):
        return "standard_vm_property"
    if any(word in lowered for word in ("token", "secret", "password", "credential", "certificate", "privatekey")):
        return "sensitive_name"
    if "github" in lowered or "runner" in lowered:
        return "runner_or_github"
    if "azure" in lowered or "fabric" in lowered or "goalstate" in lowered:
        return "azure_or_fabric"
    if "extension" in lowered or "provision" in lowered:
        return "extension_or_provisioning"
    if "url" in lowered or "uri" in lowered or "endpoint" in lowered:
        return "endpoint_name"
    if "network" in lowered or "address" in lowered or "gateway" in lowered:
        return "network_name"
    if "status" in lowered or "health" in lowered or "telemetry" in lowered:
        return "status_or_telemetry"
    return "other"


def ip_class(value: str) -> str | None:
    try:
        parsed = ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return None
    if parsed.is_loopback:
        return "ip_loopback"
    if parsed.is_link_local:
        return "ip_link_local"
    if parsed.is_private:
        return "ip_private"
    if parsed.is_global:
        return "ip_global"
    return "ip_other"


def url_class(value: str) -> tuple[str | None, bool]:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return None, False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None, False
    host = parsed.hostname.casefold().rstrip(".")
    if host == "168.63.129.16":
        host_type = "url_azure_wire_server"
    elif host.endswith(".blob.core.windows.net"):
        host_type = "url_azure_blob"
    elif host.endswith((".azure.com", ".azure.net", ".windows.net")):
        host_type = "url_azure"
    elif host.endswith((".github.com", ".github.net")):
        host_type = "url_github"
    else:
        host_type = "url_other"
    query_names = {name.casefold() for name, _ in urllib.parse.parse_qsl(parsed.query)}
    signed = bool(query_names & {"sig", "signature", "token", "key", "secret", "se", "sp"})
    return host_type, signed


def classify_value(value: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "primary": "text",
        "credential_shape": False,
        "url_host_class": None,
        "signed_url_shape": False,
    }
    stripped = value.strip()
    lowered = stripped.casefold()
    if not stripped:
        result["primary"] = "empty"
        return result
    if GUID_RE.fullmatch(stripped):
        result["primary"] = "guid"
        return result
    address_type = ip_class(stripped)
    if address_type:
        result["primary"] = address_type
        return result
    host_type, signed = url_class(stripped)
    if host_type:
        result["primary"] = "url"
        result["url_host_class"] = host_type
        result["signed_url_shape"] = signed
        result["credential_shape"] = signed
        return result
    if JWT_RE.fullmatch(stripped):
        result["primary"] = "jwt"
        result["credential_shape"] = True
        return result
    if stripped.startswith("-----BEGIN "):
        result["primary"] = "pem"
        result["credential_shape"] = True
        return result
    if lowered.startswith(("bearer ", "basic ")):
        result["primary"] = "authorization_header"
        result["credential_shape"] = True
        return result
    if VERSION_RE.fullmatch(stripped):
        result["primary"] = "dotted_numeric"
        return result
    if INTEGER_RE.fullmatch(stripped):
        result["primary"] = "integer"
        return result
    if stripped[:1] in {"{", "["}:
        try:
            json.loads(stripped)
            result["primary"] = "json"
        except json.JSONDecodeError:
            result["primary"] = "structured_text"
    elif stripped.startswith("<") and stripped.endswith(">"):
        result["primary"] = "xml_like"
    elif HOSTNAME_RE.fullmatch(stripped):
        result["primary"] = "hostname_or_label"
    elif (stripped.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", stripped)):
        result["primary"] = "path"
    elif len(stripped) >= 32 and BASE64ISH_RE.fullmatch(stripped) and entropy(stripped) >= 4.0:
        result["primary"] = "opaque_high_entropy"
        result["credential_shape"] = True
    elif len(stripped) >= 32 and HEX_RE.fullmatch(stripped) and entropy(stripped) >= 3.0:
        result["primary"] = "opaque_hex"
        result["credential_shape"] = True

    if re.search(r"(?i)(?:token|secret|password|signature|credential|private[_-]?key)\s*[:=]", stripped):
        result["credential_shape"] = True
    if re.search(r"(?i)https?://", stripped):
        result["embedded_url"] = True
        if re.search(r"(?i)[?&](?:sig|token|key|secret|se|sp)=", stripped):
            result["credential_shape"] = True
            result["signed_url_shape"] = True
    return result


def length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 8:
        return "1-8"
    if length <= 32:
        return "9-32"
    if length <= 128:
        return "33-128"
    if length <= 512:
        return "129-512"
    return "513-2047"


def inspect_pool(index: int) -> dict[str, Any]:
    path = ROOT / f".kvp_pool_{index}"
    output: dict[str, Any] = {
        "present": path.is_file(),
        "record_count": 0,
        "key_class_counts": {},
        "value_type_counts": {},
        "length_bucket_counts": {},
        "url_host_class_counts": {},
        "credential_shape_count": 0,
        "signed_url_shape_count": 0,
        "embedded_url_count": 0,
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

    key_counts: collections.Counter[str] = collections.Counter()
    value_counts: collections.Counter[str] = collections.Counter()
    length_counts: collections.Counter[str] = collections.Counter()
    host_counts: collections.Counter[str] = collections.Counter()
    for offset in range(0, len(data), RECORD_SIZE):
        record = data[offset : offset + RECORD_SIZE]
        key = c_string(record[:KEY_SIZE])
        if not key:
            continue
        value = c_string(record[KEY_SIZE:])
        output["record_count"] += 1
        key_counts[key_class(key)] += 1
        classified = classify_value(value)
        value_counts[str(classified["primary"])] += 1
        length_counts[length_bucket(len(value))] += 1
        if classified.get("url_host_class"):
            host_counts[str(classified["url_host_class"])] += 1
        output["credential_shape_count"] += int(bool(classified["credential_shape"]))
        output["signed_url_shape_count"] += int(bool(classified["signed_url_shape"]))
        output["embedded_url_count"] += int(bool(classified.get("embedded_url")))
    output["key_class_counts"] = dict(sorted(key_counts.items()))
    output["value_type_counts"] = dict(sorted(value_counts.items()))
    output["length_bucket_counts"] = dict(sorted(length_counts.items()))
    output["url_host_class_counts"] = dict(sorted(host_counts.items()))
    return output


def main() -> int:
    pools = {str(index): inspect_pool(index) for index in range(5)}
    result = {
        "schema": "meta-lhe-fairchem-kvp-value-types-v1",
        "scope": "aggregate-only local KVP key/value shape classification",
        "pools": pools,
        "credential_shape_total": sum(
            int(pool["credential_shape_count"]) for pool in pools.values()
        ),
        "signed_url_shape_total": sum(
            int(pool["signed_url_shape_count"]) for pool in pools.values()
        ),
        "raw_key_emitted": False,
        "raw_value_emitted": False,
        "hostname_address_url_or_identifier_emitted": False,
        "network_request_sent": False,
        "vmbus_request_sent": False,
        "external_state_changed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
