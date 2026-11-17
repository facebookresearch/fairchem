"""Meta LHE: bounded live Azure guest-agent/HCA authority classifier.

This probe is intentionally read-only.  It reads the disposable runner's local
HCA settings, performs bounded GET requests to the VM-local Azure WireServer
goal-state URLs, and reads only the two exact blob URLs that the current VM's
ExtensionsConfig explicitly grants read access to.  It never prints response
values, URLs, identifiers, signatures, tokens, hashes of secrets, or bodies.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import ipaddress
import json
import os
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


MARKER = "META_LHE_FAIRCHEM_T4_LIVE"
WIRESERVER = "http://168.63.129.16/machine/?comp=goalstate"
WIRE_LIMIT = 256 * 1024
BLOB_LIMIT = 512 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,95}$")
SENSITIVE_NAME = re.compile(
    r"(?:auth|bearer|credential|key|password|private|sas|secret|signature|token)",
    re.IGNORECASE,
)
JWT_SHAPE = re.compile(
    r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$"
)
BASE64_SHAPE = re.compile(r"^[A-Za-z0-9+/]{80,}={0,2}$")
PRIVATE_V4 = re.compile(
    rb"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|"
    rb"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"
)
NETWORK_TERMS = {
    "vnet": re.compile(rb"\bvnet\b|virtual[_ -]?network", re.IGNORECASE),
    "subnet": re.compile(rb"\bsubnet\b", re.IGNORECASE),
    "private_link": re.compile(rb"privatelink|private[_ -]?link", re.IGNORECASE),
    "network_profile": re.compile(rb"network[_ -]?profile", re.IGNORECASE),
    "managed_identity": re.compile(rb"managed[_ -]?identity", re.IGNORECASE),
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)
REQUEST_COUNT = 0


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    elif isinstance(value, (list, tuple, set)):
        rendered = ",".join(str(item) for item in value)
    else:
        rendered = str(value)
    print(f"{MARKER}_{name}={rendered}", flush=True)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def safe_names(values: list[str] | set[str]) -> list[str]:
    return sorted({value for value in values if SAFE_NAME.fullmatch(value)})[:96]


def size_bucket(size: int) -> str:
    if size == 0:
        return "zero"
    if size <= 1024:
        return "1_to_1024"
    if size <= 16384:
        return "1025_to_16384"
    if size <= 131072:
        return "16385_to_131072"
    return "over_131072"


def host_class(hostname: str | None) -> str:
    if not hostname:
        return "none"
    lowered = hostname.casefold().rstrip(".")
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        address = None
    if address is not None:
        if str(address) == "168.63.129.16":
            return "azure_wire_server"
        if address.is_private:
            return "private_ip"
        return "public_ip"
    if lowered.endswith(".blob.core.windows.net"):
        return "azure_public_blob"
    if lowered.endswith(".blob.storage.azure.net"):
        return "azure_storage_blob"
    if lowered.endswith((".actions.githubusercontent.com", ".actions.githubusercontent.com.")):
        return "github_actions"
    if lowered.endswith(("github.com", "githubusercontent.com")):
        return "github_public"
    return "other_dns"


def validate_wire_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(value)
        return (
            parsed.scheme == "http"
            and parsed.hostname == "168.63.129.16"
            and parsed.port in (None, 80)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError:
        return False


def validate_blob_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(value)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and "sig" in query
            and "sp" in query
            and "sr" in query
        )
    except ValueError:
        return False


def fetch(
    url: str,
    limit: int,
    *,
    wire: bool,
    retries: int = 1,
) -> tuple[int | str, bytes, str]:
    global REQUEST_COUNT
    if wire:
        if not validate_wire_url(url):
            raise ValueError("refused non-WireServer dependent URL")
        headers = {
            "x-ms-agent-name": "WALinuxAgent",
            "x-ms-version": "2012-11-30",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "User-Agent": "meta-lhe-fairchem-live-boundary/1",
        }
    else:
        if not validate_blob_url(url):
            raise ValueError("refused non-SAS blob URL")
        headers = {
            "Accept-Encoding": "identity",
            "Connection": "close",
            "User-Agent": "meta-lhe-fairchem-live-boundary/1",
            "x-ms-version": "2021-12-02",
        }

    last_error: Exception | None = None
    for attempt in range(retries):
        REQUEST_COUNT += 1
        request = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with OPENER.open(request, timeout=4) as response:
                body = response.read(limit + 1)
                return response.status, body, response.headers.get_content_type()
        except urllib.error.HTTPError as error:
            try:
                body = error.read(limit + 1)
                content_type = error.headers.get_content_type()
            finally:
                error.close()
            return error.code, body, content_type
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1)
    return "transport_error", b"", type(last_error).__name__ if last_error else "unknown"


def exact_text(root: ET.Element, name: str) -> str | None:
    values = [
        (element.text or "").strip()
        for element in root.iter()
        if local_name(element.tag) == name and (element.text or "").strip()
    ]
    return values[0] if len(values) == 1 else None


def parse_query(value: str) -> dict[str, list[str]]:
    try:
        return urllib.parse.parse_qs(
            urllib.parse.urlsplit(value).query, keep_blank_values=True
        )
    except ValueError:
        return {}


def classify_sas(label: str, value: str) -> urllib.parse.SplitResult | None:
    try:
        parsed = urllib.parse.urlsplit(value)
        query = parse_query(value)
    except ValueError:
        emit(f"{label}_parseable", False)
        return None
    segments = [
        urllib.parse.unquote(segment)
        for segment in parsed.path.split("/")
        if segment
    ]
    emit(f"{label}_parseable", True)
    emit(f"{label}_scheme", parsed.scheme)
    emit(f"{label}_host_class", host_class(parsed.hostname))
    emit(f"{label}_path_segment_count", len(segments))
    emit(f"{label}_path_segment_lengths", [len(segment) for segment in segments])
    emit(f"{label}_query_keys", safe_names(set(query)))
    permissions = "".join(query.get("sp", []))
    emit(
        f"{label}_permissions",
        permissions if re.fullmatch(r"[racwdlxtmeopi]{0,20}", permissions) else "opaque",
    )
    resource = "".join(query.get("sr", []))
    emit(
        f"{label}_resource",
        resource if re.fullmatch(r"[bcdfs]{0,4}", resource) else "opaque",
    )
    emit(f"{label}_storage_key_selector_present", "sk" in query)
    expiration_year = "unparseable"
    if len(query.get("se", [])) == 1:
        try:
            expiration_year = dt.datetime.fromisoformat(
                query["se"][0].replace("Z", "+00:00")
            ).year
        except ValueError:
            pass
    emit(f"{label}_expiration_year", expiration_year)
    return parsed


def value_shape(value: str) -> str:
    if not value:
        return "empty"
    if JWT_SHAPE.fullmatch(value):
        return "jwt"
    try:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            return "sas_url" if "sig" in parse_query(value) else "url"
    except ValueError:
        pass
    compact = "".join(value.split())
    if len(compact) <= 65536 and BASE64_SHAPE.fullmatch(compact):
        try:
            base64.b64decode(compact, validate=True)
            return "base64"
        except (binascii.Error, ValueError):
            pass
    return "opaque"


def classify_xml(label: str, body: bytes) -> ET.Element | None:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        emit(f"{label}_xml_parsed", False)
        return None
    elements = list(root.iter())
    tags = safe_names([local_name(element.tag) for element in elements])
    attributes = safe_names(
        [local_name(name) for element in elements for name in element.attrib]
    )
    texts = [(element.text or "").strip() for element in elements]
    nonempty = [value for value in texts if value]
    shapes = Counter(value_shape(value) for value in nonempty)
    emit(f"{label}_xml_parsed", True)
    emit(f"{label}_root", local_name(root.tag) if SAFE_NAME.fullmatch(local_name(root.tag)) else "unsafe")
    emit(f"{label}_element_count", len(elements))
    emit(f"{label}_tag_names", tags)
    emit(f"{label}_attribute_names", attributes)
    emit(f"{label}_nonempty_text_count", len(nonempty))
    emit(
        f"{label}_sensitive_name_count",
        sum(bool(SENSITIVE_NAME.search(name)) for name in tags + attributes),
    )
    for shape in ("url", "sas_url", "jwt", "base64", "opaque"):
        emit(f"{label}_{shape}_text_count", shapes[shape])
    emit(f"{label}_private_ipv4_shape_count", len(PRIVATE_V4.findall(body)))
    for term, pattern in NETWORK_TERMS.items():
        emit(f"{label}_{term}_term_count", len(pattern.findall(body)))
    return root


def walk_json(value: object) -> tuple[list[str], list[str], int, int]:
    keys: list[str] = []
    strings: list[str] = []
    objects = 0
    arrays = 0
    stack = [value]
    while stack and len(keys) + len(strings) < 20000:
        item = stack.pop()
        if isinstance(item, dict):
            objects += 1
            for key, child in item.items():
                if isinstance(key, str):
                    keys.append(key)
                stack.append(child)
        elif isinstance(item, list):
            arrays += 1
            stack.extend(item)
        elif isinstance(item, str):
            strings.append(item)
    return keys, strings, objects, arrays


def classify_blob_body(label: str, status: int | str, body: bytes, content_type: str) -> None:
    emit(f"{label}_http_status", status)
    emit(f"{label}_size_within_limit", len(body) <= BLOB_LIMIT)
    emit(f"{label}_size_bucket", size_bucket(min(len(body), BLOB_LIMIT + 1)))
    emit(
        f"{label}_content_type_class",
        "json" if "json" in content_type else "xml" if "xml" in content_type else "other",
    )
    if not isinstance(status, int) or status < 200 or status >= 300 or len(body) > BLOB_LIMIT:
        emit(f"{label}_parsed", False)
        return
    try:
        document = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        document = None
    if document is not None:
        keys, strings, objects, arrays = walk_json(document)
        shapes = Counter(value_shape(value) for value in strings)
        emit(f"{label}_parsed", True)
        emit(f"{label}_format", "json")
        emit(f"{label}_key_names", safe_names(keys))
        emit(f"{label}_object_count", objects)
        emit(f"{label}_array_count", arrays)
        emit(f"{label}_string_count", len(strings))
        emit(f"{label}_sensitive_key_name_count", sum(bool(SENSITIVE_NAME.search(key)) for key in keys))
        for shape in ("url", "sas_url", "jwt", "base64"):
            emit(f"{label}_{shape}_value_count", shapes[shape])
        return
    root = classify_xml(label, body)
    emit(f"{label}_parsed", root is not None)
    if root is not None:
        emit(f"{label}_format", "xml")


def classify_jwt(label: str, token: str) -> None:
    pieces = token.split(".")
    if len(pieces) != 3:
        emit(f"{label}_jwt_parseable", False)
        return
    try:
        payload_piece = pieces[1] + "=" * (-len(pieces[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_piece))
    except (ValueError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        emit(f"{label}_jwt_parseable", False)
        return
    if not isinstance(payload, dict):
        emit(f"{label}_jwt_parseable", False)
        return
    emit(f"{label}_jwt_parseable", True)
    emit(f"{label}_claim_names", safe_names(set(payload)))
    exp = payload.get("exp")
    iat = payload.get("iat")
    lifetime = "unknown"
    if isinstance(exp, (int, float)) and isinstance(iat, (int, float)):
        minutes = max(0, int((exp - iat) / 60))
        lifetime = "under_1h" if minutes <= 60 else "1h_to_24h" if minutes <= 1440 else "over_24h"
    emit(f"{label}_lifetime_bucket", lifetime)
    scope = payload.get("scp")
    scope_values = scope.split() if isinstance(scope, str) else scope if isinstance(scope, list) else []
    scope_families: set[str] = set()
    rejected = 0
    for item in scope_values:
        if not isinstance(item, str):
            rejected += 1
            continue
        family = item.split(":", 1)[0]
        if SAFE_NAME.fullmatch(family) and len(family) <= 64:
            scope_families.add(family)
        else:
            rejected += 1
    emit(f"{label}_scope_family_names", sorted(scope_families))
    emit(f"{label}_scope_value_count", len(scope_values))
    emit(f"{label}_scope_rejected_count", rejected)


def classify_hca() -> tuple[urllib.parse.SplitResult | None, list[str]]:
    settings_path = Path("/opt/hca/.settings")
    emit("hca_settings_present", settings_path.is_file())
    if not settings_path.is_file() or settings_path.stat().st_size > 131072:
        return None, []
    try:
        settings = json.loads(settings_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        emit("hca_settings_parsed", False)
        return None, []
    if not isinstance(settings, dict):
        emit("hca_settings_parsed", False)
        return None, []
    emit("hca_settings_parsed", True)
    emit("hca_settings_mode", format(stat.S_IMODE(settings_path.stat().st_mode), "o"))
    emit("hca_setting_keys", safe_names(set(settings)))
    diagnostics = settings.get("diagnosticsSasUri")
    diagnostics_parts = None
    if isinstance(diagnostics, str):
        diagnostics_parts = classify_sas("hca_diagnostics_sas", diagnostics)
    token = settings.get("authToken")
    if isinstance(token, str):
        classify_jwt("hca_auth", token)
    scheduler = settings.get("schedulerApiUrl")
    if isinstance(scheduler, str):
        try:
            scheduler_parts = urllib.parse.urlsplit(scheduler)
            emit("hca_scheduler_scheme", scheduler_parts.scheme)
            emit("hca_scheduler_host_class", host_class(scheduler_parts.hostname))
            emit("hca_scheduler_path_segment_count", len([p for p in scheduler_parts.path.split("/") if p]))
            emit("hca_scheduler_query_keys", safe_names(set(parse_query(scheduler))))
        except ValueError:
            emit("hca_scheduler_parseable", False)
    identifiers = [
        value
        for key, value in settings.items()
        if isinstance(value, str)
        and key not in {"authToken", "diagnosticsSasUri", "schedulerApiUrl", "traceApiUrl"}
        and len(value) <= 256
    ]
    return diagnostics_parts, identifiers


def main() -> None:
    diagnostics_parts, hca_identifiers = classify_hca()

    goal_status, goal_body, goal_type = fetch(
        WIRESERVER, WIRE_LIMIT, wire=True, retries=10
    )
    emit("goalstate_http_status", goal_status)
    emit("goalstate_size_within_limit", len(goal_body) <= WIRE_LIMIT)
    if goal_status != 200 or len(goal_body) > WIRE_LIMIT:
        emit("goalstate_parsed", False)
        emit("network_requests", REQUEST_COUNT)
        emit("http_methods", "GET_only")
        emit("state_changed", False)
        return
    goal = classify_xml("goalstate", goal_body)
    goal_body = b""
    if goal is None:
        emit("network_requests", REQUEST_COUNT)
        emit("http_methods", "GET_only")
        emit("state_changed", False)
        return

    identity_values = [
        value
        for name in ("ContainerId", "InstanceId", "ConfigName")
        if (value := exact_text(goal, name)) is not None
    ]
    dependent: dict[str, str] = {}
    for name in (
        "HostingEnvironmentConfig",
        "SharedConfig",
        "ExtensionsConfig",
        "Certificates",
        "RemoteAccessInfo",
    ):
        value = exact_text(goal, name)
        emit(f"goalstate_{name.lower()}_present", value is not None)
        emit(f"goalstate_{name.lower()}_vm_local", value is not None and validate_wire_url(value))
        if value is not None and validate_wire_url(value):
            dependent[name] = value

    documents: dict[str, ET.Element] = {}
    for name, url in dependent.items():
        status, body, content_type = fetch(url, WIRE_LIMIT, wire=True, retries=4)
        label = "wire_" + name.lower()
        emit(f"{label}_http_status", status)
        emit(f"{label}_size_within_limit", len(body) <= WIRE_LIMIT)
        if status == 200 and len(body) <= WIRE_LIMIT:
            parsed = classify_xml(label, body)
            if parsed is not None:
                documents[name] = parsed
        body = b""

    extensions = documents.get("ExtensionsConfig")
    sas_values: dict[str, str] = {}
    sas_parts: dict[str, urllib.parse.SplitResult] = {}
    if extensions is not None:
        for name in (
            "Plugin",
            "RuntimeSettings",
            "ExtensionRuntimeSettings",
            "Settings",
            "ProtectedSettings",
        ):
            emit(
                f"extensions_{name.lower()}_count",
                sum(local_name(element.tag) == name for element in extensions.iter()),
            )
        for label, name in (
            ("status_blob_sas", "StatusUploadBlob"),
            ("artifact_profile_sas", "InVMArtifactsProfileBlob"),
        ):
            value = exact_text(extensions, name)
            emit(f"{label}_present", value is not None)
            emit(f"{label}_validated", value is not None and validate_blob_url(value))
            if value is not None and validate_blob_url(value):
                sas_values[label] = value
                parts = classify_sas(label, value)
                if parts is not None:
                    sas_parts[label] = parts

    if "status_blob_sas" in sas_parts and "artifact_profile_sas" in sas_parts:
        emit(
            "waagent_blob_authority_equal",
            sas_parts["status_blob_sas"].netloc == sas_parts["artifact_profile_sas"].netloc,
        )
        emit(
            "waagent_blob_path_equal",
            sas_parts["status_blob_sas"].path == sas_parts["artifact_profile_sas"].path,
        )

    if diagnostics_parts is not None:
        for label, parts in sas_parts.items():
            emit(
                f"hca_diagnostics_{label}_authority_equal",
                diagnostics_parts.netloc == parts.netloc,
            )
            emit(
                f"hca_diagnostics_{label}_path_equal",
                diagnostics_parts.path == parts.path,
            )

    comparison_values = identity_values + hca_identifiers
    for label, parts in sas_parts.items():
        decoded_segments = {
            urllib.parse.unquote(segment)
            for segment in parts.path.split("/")
            if segment
        }
        emit(
            f"{label}_path_matches_current_vm_identifier_count",
            len(decoded_segments.intersection(comparison_values)),
        )

    for label, value in sas_values.items():
        permissions = "".join(parse_query(value).get("sp", []))
        if "r" not in permissions:
            emit(f"{label}_get_skipped_no_read", True)
            continue
        status, body, content_type = fetch(value, BLOB_LIMIT, wire=False)
        classify_blob_body(label.removesuffix("_sas") + "_body", status, body, content_type)
        body = b""

    certificates = documents.get("Certificates")
    if certificates is not None:
        formats = [
            (element.text or "").strip()
            for element in certificates.iter()
            if local_name(element.tag) == "Format" and (element.text or "").strip()
        ]
        data_values = [
            "".join((element.text or "").split())
            for element in certificates.iter()
            if local_name(element.tag) == "Data" and (element.text or "").strip()
        ]
        emit("certificates_format_count", len(formats))
        emit("certificates_pkcs7_pfx_format_count", sum(value == "Pkcs7BlobWithPfxContents" for value in formats))
        emit("certificates_data_count", len(data_values))
        decoded_sizes: list[int] = []
        failures = 0
        for value in data_values:
            try:
                decoded_sizes.append(len(base64.b64decode(value, validate=True)))
            except (binascii.Error, ValueError):
                failures += 1
        emit("certificates_data_decode_failures", failures)
        emit("certificates_decoded_size_buckets", sorted(Counter(size_bucket(size) for size in decoded_sizes).items()))

    waagent = Path("/var/lib/waagent")
    key_candidates: list[Path] = []
    certificate_candidates: list[Path] = []
    if waagent.is_dir():
        for path in waagent.rglob("*"):
            try:
                if not path.is_file() or len(path.relative_to(waagent).parts) > 4:
                    continue
            except OSError:
                continue
            if path.suffix.casefold() in (".prv", ".key", ".pem") and "private" in path.name.casefold():
                key_candidates.append(path)
            if path.suffix.casefold() in (".crt", ".cer", ".pem") and "cert" in path.name.casefold():
                certificate_candidates.append(path)
    emit("waagent_private_key_candidate_count", len(key_candidates))
    emit("waagent_certificate_candidate_count", len(certificate_candidates))
    emit("waagent_private_key_mode_counts", sorted(Counter(format(stat.S_IMODE(path.stat().st_mode), "o") for path in key_candidates).items()))
    emit("waagent_certificate_mode_counts", sorted(Counter(format(stat.S_IMODE(path.stat().st_mode), "o") for path in certificate_candidates).items()))

    emit("network_requests", REQUEST_COUNT)
    emit("http_methods", "GET_only")
    emit("redirects_followed", 0)
    emit("raw_url_or_identifier_emitted", False)
    emit("raw_token_signature_or_body_emitted", False)
    emit("filesystem_changes", 0)
    emit("process_changes", 0)
    emit("state_changed", False)


if __name__ == "__main__":
    main()
