"""Bounded Azure IMDS managed-identity classifier for Meta LHE.

The probe sends two direct, read-only requests to the current disposable VM's
IMDS identity endpoints. It emits only HTTP status, JSON key names and coarse
error/claim classes. If and only if IMDS issues a management token, it performs
bounded ARM reads and emits aggregate counts without resource identifiers.
No response body, credential, address, hostname or identifier is printed.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import ssl
import sys
import time
from urllib.parse import quote


MARKER = "META_LHE_FAIRCHEM_IMDS_IDENTITY"
MAX_BODY = 4 * 1024 * 1024
KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    elif isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        rendered = str(value)
    print(f"{MARKER}_{name}={rendered}", flush=True)


def safe_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(
        key for key in value if isinstance(key, str) and KEY_PATTERN.fullmatch(key)
    )


def case_get(value: object, name: str) -> object | None:
    if not isinstance(value, dict):
        return None
    for key, child in value.items():
        if isinstance(key, str) and key.casefold() == name.casefold():
            return child
    return None


def read_response(response: http.client.HTTPResponse) -> bytes:
    declared = response.getheader("Content-Length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY:
        raise ValueError("response too large")
    body = response.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise ValueError("response too large")
    return body


def imds_get(path: str) -> tuple[int | None, bytes, str]:
    connection = http.client.HTTPConnection("169.254.169.254", 80, timeout=5)
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Metadata": "true",
                "Connection": "close",
                "User-Agent": "Meta-LHE-FairChem-IMDS-Boundary/1.0",
            },
        )
        response = connection.getresponse()
        content_type = response.getheader("Content-Type", "")
        body = read_response(response)
        return response.status, body, content_type
    except (OSError, http.client.HTTPException, ValueError):
        return None, b"", ""
    finally:
        connection.close()


def arm_get(path: str, token: str) -> tuple[int | None, bytes, str]:
    connection = http.client.HTTPSConnection(
        "management.azure.com", 443, timeout=8, context=ssl.create_default_context()
    )
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/json",
                "Connection": "close",
                "User-Agent": "Meta-LHE-FairChem-ARM-Boundary/1.0",
            },
        )
        response = connection.getresponse()
        content_type = response.getheader("Content-Type", "")
        body = read_response(response)
        return response.status, body, content_type
    except (OSError, ssl.SSLError, http.client.HTTPException, ValueError):
        return None, b"", ""
    finally:
        connection.close()


def parse_json(body: bytes) -> object | None:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def decode_jwt_payload(token: str) -> dict[str, object] | None:
    pieces = token.split(".")
    if len(pieces) != 3 or not all(pieces):
        return None
    try:
        encoded = pieces[1].encode("ascii")
        encoded += b"=" * ((4 - len(encoded) % 4) % 4)
        value = json.loads(base64.urlsafe_b64decode(encoded))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def error_flags(body: bytes) -> dict[str, bool]:
    text = body.decode("utf-8", "replace").casefold()
    return {
        "identity_not_found": any(
            marker in text
            for marker in (
                "identity not found",
                "no managed identity",
                "no identity assigned",
                "identity is not assigned",
            )
        ),
        "multiple_identities": "multiple" in text
        and ("identity" in text or "identities" in text),
        "selector_required": any(
            marker in text
            for marker in ("client_id", "clientid", "resource_id", "mi_res_id")
        )
        and any(marker in text for marker in ("specify", "required", "provide")),
        "identity_endpoint_disabled": "identity" in text and "disabled" in text,
    }


def main() -> int:
    emit("schema", "v1")
    emit("effective_uid_zero", os.geteuid() == 0)

    info_status, info_body, info_content_type = imds_get(
        "/metadata/identity/info?api-version=2021-02-01"
    )
    info = parse_json(info_body)
    emit(
        "identity_info_status",
        "transport_error" if info_status is None else info_status,
    )
    emit("identity_info_json", isinstance(info, dict))
    emit("identity_info_json_keys", safe_keys(info))
    emit("identity_info_content_type_json", "json" in info_content_type.casefold())
    emit("identity_info_client_id_present", case_get(info, "client_id") is not None)
    emit("identity_info_object_id_present", case_get(info, "object_id") is not None)
    emit(
        "identity_info_resource_id_present",
        case_get(info, "msi_res_id") is not None
        or case_get(info, "mi_res_id") is not None
        or case_get(info, "resource_id") is not None,
    )
    emit("identity_info_error_flags", error_flags(info_body))

    resource = quote("https://management.azure.com/", safe="")
    token_status, token_body, token_content_type = imds_get(
        "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=" + resource
    )
    token_document = parse_json(token_body)
    emit("token_status", "transport_error" if token_status is None else token_status)
    emit("token_response_json", isinstance(token_document, dict))
    emit("token_response_json_keys", safe_keys(token_document))
    emit("token_content_type_json", "json" in token_content_type.casefold())
    emit("token_error_flags", error_flags(token_body))

    token = case_get(token_document, "access_token")
    token_type = case_get(token_document, "token_type")
    emit("access_token_present", isinstance(token, str) and bool(token))
    emit(
        "token_type_bearer",
        isinstance(token_type, str) and token_type.casefold() == "bearer",
    )
    arm_request_count = 0
    if not isinstance(token, str) or not token:
        emit("arm_request_count", arm_request_count)
        emit("raw_response_bodies_emitted", 0)
        emit("raw_credentials_or_identifiers_emitted", 0)
        return 0

    payload = decode_jwt_payload(token)
    emit("token_jwt", payload is not None)
    emit("token_claim_names", safe_keys(payload))
    if isinstance(payload, dict):
        emit(
            "token_audience_is_arm",
            case_get(payload, "aud") in (
                "https://management.azure.com/",
                "https://management.core.windows.net/",
            ),
        )
        issued_at = case_get(payload, "iat")
        expires_at = case_get(payload, "exp")
        now = int(time.time())
        emit(
            "token_expiry_class",
            "future_over_1h"
            if isinstance(expires_at, int) and expires_at - now > 3600
            else "future_under_1h"
            if isinstance(expires_at, int) and expires_at > now
            else "expired_or_unknown",
        )
        emit(
            "token_lifetime_present",
            isinstance(issued_at, int) and isinstance(expires_at, int),
        )

    subscriptions_status, subscriptions_body, _ = arm_get(
        "/subscriptions?api-version=2020-01-01", token
    )
    arm_request_count += 1
    subscriptions_document = parse_json(subscriptions_body)
    subscriptions = case_get(subscriptions_document, "value")
    if not isinstance(subscriptions, list):
        subscriptions = []
    emit(
        "arm_subscriptions_status",
        "transport_error" if subscriptions_status is None else subscriptions_status,
    )
    emit("arm_subscriptions_json_keys", safe_keys(subscriptions_document))
    emit("arm_subscription_count_first_page", len(subscriptions))
    emit(
        "arm_subscriptions_next_link_present",
        case_get(subscriptions_document, "nextLink") is not None,
    )

    status_counts: dict[str, dict[str, int]] = {
        "resources": {},
        "role_assignments": {},
    }
    tested_subscriptions = 0
    for item in subscriptions[:3]:
        subscription_id = case_get(item, "subscriptionId")
        if not isinstance(subscription_id, str) or not UUID_PATTERN.fullmatch(
            subscription_id
        ):
            continue
        tested_subscriptions += 1
        encoded_id = quote(subscription_id, safe="")
        for label, path in (
            (
                "resources",
                f"/subscriptions/{encoded_id}/resources?api-version=2021-04-01&$top=1",
            ),
            (
                "role_assignments",
                f"/subscriptions/{encoded_id}/providers/Microsoft.Authorization/"
                "roleAssignments?api-version=2022-04-01&$filter=atScope()&$top=1",
            ),
        ):
            status, body, _ = arm_get(path, token)
            arm_request_count += 1
            key = "transport_error" if status is None else str(status)
            status_counts[label][key] = status_counts[label].get(key, 0) + 1
            document = parse_json(body)
            emit(
                f"arm_{label}_response_{tested_subscriptions}_value_present",
                isinstance(case_get(document, "value"), list),
            )
    emit("arm_tested_subscription_count", tested_subscriptions)
    emit("arm_status_counts", status_counts)
    emit("arm_request_count", arm_request_count)
    emit("raw_response_bodies_emitted", 0)
    emit("raw_credentials_or_identifiers_emitted", 0)
    token = ""
    return 0


if __name__ == "__main__":
    sys.exit(main())
