#!/usr/bin/env python3
"""Aggregate-only classifier for cloud-init data in Hyper-V KVP pool 1."""

from __future__ import annotations

import base64
import collections
import json
import re
import zlib
from pathlib import Path
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_KVP_COMPRESSED"
POOL = Path("/var/lib/hyperv/.kvp_pool_1")
KEY_SIZE = 512
VALUE_SIZE = 2048
RECORD_SIZE = KEY_SIZE + VALUE_SIZE
MAX_RECORDS = 4096
MAX_MESSAGE_BYTES = 512 * 1024
MAX_DECOMPRESSED_BYTES = 1024 * 1024
EXPECTED_FIELDS = {"name", "type", "ts", "result", "duration", "msg", "msg_i"}
REQUIRED_FIELDS = {"name", "type", "ts", "msg"}
GUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
KEY_RE = re.compile(
    rf"^CLOUD_INIT\|[0-9]+\|[^|\x00]+\|[^|\x00]+\|{GUID}\|{GUID}(?:\|[0-9]+)?$"
)
MATERIAL_PATTERNS = {
    "aws_access_key": re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "github_token": re.compile(rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "jwt": re.compile(rb"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])"),
    "pem_private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "authorization_bearer": re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "signed_url": re.compile(rb"(?i)https?://[^\s\"']{1,2048}[?&](?:sig|signature|token|key|secret)=[^&\s\"']+"),
}


def c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def material_counts(data: bytes) -> dict[str, int]:
    return {
        name: len(pattern.findall(data))
        for name, pattern in MATERIAL_PATTERNS.items()
        if pattern.search(data)
    }


def bounded_zlib_decompress(data: bytes) -> tuple[bytes | None, str]:
    try:
        decoder = zlib.decompressobj()
        output = decoder.decompress(data, MAX_DECOMPRESSED_BYTES + 1)
        if len(output) > MAX_DECOMPRESSED_BYTES:
            return None, "size_limit"
        output += decoder.flush(MAX_DECOMPRESSED_BYTES + 1 - len(output))
    except zlib.error:
        return None, "zlib_error"
    if len(output) > MAX_DECOMPRESSED_BYTES:
        return None, "size_limit"
    if not decoder.eof or decoder.unused_data:
        return None, "stream_boundary_error"
    return output, "ok"


def read_records() -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    counters: collections.Counter[str] = collections.Counter()
    if not POOL.is_file():
        counters["pool_missing"] += 1
        return [], dict(counters)
    try:
        size = POOL.stat().st_size
        if size % RECORD_SIZE or size > MAX_RECORDS * RECORD_SIZE:
            counters["pool_size_refused"] += 1
            return [], dict(counters)
        raw = POOL.read_bytes()
    except OSError:
        counters["pool_read_error"] += 1
        return [], dict(counters)

    records: list[tuple[str, dict[str, Any]]] = []
    for offset in range(0, len(raw), RECORD_SIZE):
        record = raw[offset : offset + RECORD_SIZE]
        key = c_string(record[:KEY_SIZE])
        if not key:
            continue
        counters["record_count"] += 1
        counters["cloud_init_key_format_count"] += int(bool(KEY_RE.fullmatch(key)))
        value = c_string(record[KEY_SIZE:])
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            counters["json_parse_failure_count"] += 1
            continue
        if not isinstance(parsed, dict):
            counters["json_non_object_count"] += 1
            continue
        counters["json_object_count"] += 1
        counters["official_schema_count"] += int(
            REQUIRED_FIELDS.issubset(parsed) and set(parsed).issubset(EXPECTED_FIELDS)
        )
        counters["unknown_field_total"] += len(set(parsed) - EXPECTED_FIELDS)
        records.append((key, parsed))
    return records, dict(counters)


def classify() -> dict[str, Any]:
    records, read_counts = read_records()
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    group_consistency_failures = 0
    for key, value in records:
        message_index = value.get("msg_i")
        if isinstance(message_index, int) and re.search(r"\|[0-9]+$", key):
            base_key = key.rsplit("|", 1)[0]
        else:
            base_key = key
        groups[base_key].append(value)

    group_counts: collections.Counter[str] = collections.Counter()
    material: collections.Counter[str] = collections.Counter()
    compressed_sizes: list[int] = []
    decompressed_sizes: list[int] = []
    for items in groups.values():
        items.sort(key=lambda item: item.get("msg_i", 0))
        event_types = {item.get("type") for item in items}
        event_names = {item.get("name") for item in items}
        if len(event_types) != 1 or len(event_names) != 1:
            group_consistency_failures += 1
            continue
        fragments = [item.get("msg") for item in items]
        if not all(isinstance(fragment, str) for fragment in fragments):
            group_counts["non_string_message"] += 1
            continue
        message = "".join(fragments)
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            group_counts["message_size_refused"] += 1
            continue
        event_type = next(iter(event_types))
        event_name = next(iter(event_names))
        if event_type != "compressed":
            group_counts["plain_event"] += 1
            material.update(material_counts(message.encode("utf-8")))
            continue

        group_counts["compressed_event"] += 1
        group_counts["compressed_dmesg_event"] += int(event_name == "dmesg")
        group_counts["compressed_other_name_event"] += int(event_name != "dmesg")
        try:
            envelope = json.loads(message)
        except json.JSONDecodeError:
            group_counts["compressed_envelope_parse_failure"] += 1
            continue
        if not isinstance(envelope, dict) or set(envelope) != {"encoding", "data"}:
            group_counts["compressed_envelope_schema_failure"] += 1
            continue
        group_counts["gz_b64_encoding"] += int(envelope.get("encoding") == "gz+b64")
        encoded = envelope.get("data")
        if envelope.get("encoding") != "gz+b64" or not isinstance(encoded, str):
            group_counts["compressed_encoding_failure"] += 1
            continue
        try:
            compact = re.sub(rb"\s+", b"", encoded.encode("ascii"))
            compressed = base64.b64decode(compact, validate=True)
        except (UnicodeEncodeError, ValueError):
            group_counts["base64_decode_failure"] += 1
            continue
        compressed_sizes.append(len(compressed))
        decoded, status = bounded_zlib_decompress(compressed)
        group_counts[f"zlib_{status}"] += 1
        if decoded is None:
            continue
        decompressed_sizes.append(len(decoded))
        group_counts["decompressed_utf8"] += int(
            decoded.decode("utf-8", errors="replace").encode("utf-8") == decoded
        )
        material.update(material_counts(decoded))

    return {
        "schema": "meta-lhe-fairchem-kvp-compressed-class-v1",
        "scope": "aggregate-only cloud-init KVP schema and compressed-payload classification",
        "read_counts": read_counts,
        "group_count": len(groups),
        "group_consistency_failure_count": group_consistency_failures,
        "group_class_counts": dict(sorted(group_counts.items())),
        "credential_material_pattern_counts": dict(sorted(material.items())),
        "credential_material_pattern_total": sum(material.values()),
        "compressed_size_summary": {
            "count": len(compressed_sizes),
            "min": min(compressed_sizes, default=0),
            "max": max(compressed_sizes, default=0),
        },
        "decompressed_size_summary": {
            "count": len(decompressed_sizes),
            "min": min(decompressed_sizes, default=0),
            "max": max(decompressed_sizes, default=0),
        },
        "raw_key_emitted": False,
        "raw_value_emitted": False,
        "identifier_hostname_url_or_body_emitted": False,
        "network_or_vmbus_request_sent": False,
        "external_state_changed": False,
    }


def main() -> int:
    print(f"{PREFIX}_json={json.dumps(classify(), sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
