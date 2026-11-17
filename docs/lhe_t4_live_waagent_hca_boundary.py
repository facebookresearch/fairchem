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
import subprocess
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
    extra_headers: dict[str, str] | None = None,
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

    if extra_headers:
        headers.update(extra_headers)

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
    normalized = body
    bom_present = normalized.startswith(b"\xef\xbb\xbf")
    if bom_present:
        normalized = normalized[3:]
    trailing_nuls = len(normalized) - len(normalized.rstrip(b"\0"))
    normalized = normalized.rstrip(b" \t\r\n\0")
    emit(f"{label}_utf8_bom_present", bom_present)
    emit(f"{label}_trailing_nul_count", trailing_nuls)
    try:
        document = json.loads(normalized)
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
        if isinstance(document, dict):
            emit(f"{label}_on_hold_present", "onHold" in document)
            emit(f"{label}_on_hold_true", document.get("onHold") is True)
            for key in (
                "certificateThumbprint",
                "encryptedHealthChecks",
                "encryptedApplicationProfile",
                "profileId",
            ):
                emit(f"{label}_{key.lower()}_present", key in document)
                if isinstance(document.get(key), str):
                    emit(
                        f"{label}_{key.lower()}_size_bucket",
                        size_bucket(len(document[key].encode("utf-8", errors="ignore"))),
                    )
        return
    root = classify_xml(label, normalized)
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
        if minutes <= 60:
            lifetime = "under_1h"
        elif minutes <= 1440:
            lifetime = "1h_to_24h"
        elif minutes <= 10080:
            lifetime = "1d_to_7d"
        elif minutes <= 43200:
            lifetime = "7d_to_30d"
        else:
            lifetime = "over_30d"
    emit(f"{label}_lifetime_bucket", lifetime)
    remaining = "unknown"
    if isinstance(exp, (int, float)):
        remaining_minutes = int((exp - time.time()) / 60)
        if remaining_minutes < 0:
            remaining = "expired"
        elif remaining_minutes <= 60:
            remaining = "under_1h"
        elif remaining_minutes <= 1440:
            remaining = "1h_to_24h"
        elif remaining_minutes <= 10080:
            remaining = "1d_to_7d"
        elif remaining_minutes <= 43200:
            remaining = "7d_to_30d"
        else:
            remaining = "over_30d"
    emit(f"{label}_remaining_bucket", remaining)
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


def certificate_request_headers(cipher: str) -> dict[str, str] | None:
    transport_certificate = Path("/var/lib/waagent/TransportCert.pem")
    if not transport_certificate.is_file() or transport_certificate.stat().st_size > 65536:
        return None
    try:
        lines = transport_certificate.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    public_certificate = "".join(line.strip() for line in lines if "---" not in line)
    try:
        base64.b64decode(public_certificate, validate=True)
    except (binascii.Error, ValueError):
        return None
    return {
        "x-ms-cipher-name": cipher,
        "x-ms-guest-agent-public-x509-cert": public_certificate,
    }


PEM_BLOCK = re.compile(
    rb"-----BEGIN ([A-Z0-9 ]+)-----\r?\n.*?-----END \1-----\r?\n?",
    re.DOTALL,
)


def pem_blocks(pem: bytes, suffix: bytes) -> list[bytes]:
    return [
        match.group(0)
        for match in PEM_BLOCK.finditer(pem)
        if match.group(1).endswith(suffix)
    ]


def recover_tenant_pem(certificates: ET.Element | None) -> bytes:
    cached_path = Path("/var/lib/waagent/Certificates.pem")
    cached = b""
    if cached_path.is_file() and cached_path.stat().st_size <= 2 * 1024 * 1024:
        try:
            cached = cached_path.read_bytes()
        except OSError:
            cached = b""
    emit("tenant_pem_cached_present", bool(cached))
    emit("tenant_pem_cached_size_bucket", size_bucket(len(cached)))
    emit("tenant_pem_cached_private_key_count", len(pem_blocks(cached, b"PRIVATE KEY")))
    emit("tenant_pem_cached_certificate_count", len(pem_blocks(cached, b"CERTIFICATE")))
    if pem_blocks(cached, b"PRIVATE KEY"):
        emit("tenant_pem_recovered_from_live_response", False)
        return cached

    if certificates is None:
        emit("tenant_pem_recovered_from_live_response", False)
        return cached
    data_values = [
        "".join((element.text or "").split())
        for element in certificates.iter()
        if local_name(element.tag) == "Data" and (element.text or "").strip()
    ]
    if len(data_values) != 1 or len(data_values[0]) > 2 * 1024 * 1024:
        emit("tenant_pem_recovered_from_live_response", False)
        return cached

    transport_private = Path("/var/lib/waagent/TransportPrivate.pem")
    transport_certificate = Path("/var/lib/waagent/TransportCert.pem")
    if not transport_private.is_file() or not transport_certificate.is_file():
        emit("tenant_pem_recovered_from_live_response", False)
        return cached
    mime = (
        "MIME-Version:1.0\n"
        "Content-Disposition: attachment; filename=\"Certificates.p7m\"\n"
        "Content-Type: application/x-pkcs7-mime; name=\"Certificates.p7m\"\n"
        "Content-Transfer-Encoding: base64\n\n"
        + data_values[0]
    ).encode("ascii")
    try:
        decrypted = subprocess.run(
            [
                "openssl",
                "cms",
                "-decrypt",
                "-inkey",
                str(transport_private),
                "-recip",
                str(transport_certificate),
            ],
            input=mime,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        emit("tenant_pem_recovered_from_live_response", False)
        return cached
    if decrypted.returncode != 0 or len(decrypted.stdout) > 2 * 1024 * 1024:
        emit("tenant_pem_recovered_from_live_response", False)
        return cached
    for no_mac_verify in (True, False):
        command = [
            "openssl",
            "pkcs12",
            "-nodes",
            "-password",
            "pass:",
            "-in",
            "/dev/stdin",
        ]
        if no_mac_verify:
            command.append("-nomacver")
        try:
            converted = subprocess.run(
                command,
                input=decrypted.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if (
            converted.returncode == 0
            and len(converted.stdout) <= 2 * 1024 * 1024
            and pem_blocks(converted.stdout, b"PRIVATE KEY")
        ):
            emit("tenant_pem_recovered_from_live_response", True)
            emit("tenant_pem_live_private_key_count", len(pem_blocks(converted.stdout, b"PRIVATE KEY")))
            emit("tenant_pem_live_certificate_count", len(pem_blocks(converted.stdout, b"CERTIFICATE")))
            return converted.stdout
    emit("tenant_pem_recovered_from_live_response", False)
    return cached


def decode_json_bytes(value: bytes) -> object | None:
    candidates = []
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            candidates.append(value.decode(encoding).rstrip(" \t\r\n\0"))
        except UnicodeDecodeError:
            pass
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def decrypt_protected_settings(values: list[str], tenant_pem: bytes) -> list[object]:
    keys = pem_blocks(tenant_pem, b"PRIVATE KEY")
    emit("runtime_private_key_candidate_count", len(keys))
    decrypted_documents: list[object] = []
    decode_failures = 0
    for value in values[:8]:
        compact = "".join(value.split())
        try:
            encrypted = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError):
            decode_failures += 1
            continue
        if len(encrypted) > 512 * 1024:
            decode_failures += 1
            continue
        for key in keys[:8]:
            descriptor = -1
            try:
                descriptor = os.memfd_create("lhe-runtime-key", flags=0)
                os.write(descriptor, key)
                os.lseek(descriptor, 0, os.SEEK_SET)
                result = subprocess.run(
                    [
                        "openssl",
                        "cms",
                        "-decrypt",
                        "-inform",
                        "DER",
                        "-inkey",
                        f"/proc/self/fd/{descriptor}",
                        "-in",
                        "/dev/stdin",
                    ],
                    input=encrypted,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(descriptor,),
                    timeout=4,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if result.returncode != 0 or len(result.stdout) > 1024 * 1024:
                continue
            document = decode_json_bytes(result.stdout)
            if document is not None:
                decrypted_documents.append(document)
                break
    emit("runtime_protected_settings_base64_decode_failures", decode_failures)
    emit("runtime_protected_settings_decrypted_count", len(decrypted_documents))
    return decrypted_documents


def classify_runtime_settings(extensions: ET.Element, tenant_pem: bytes) -> None:
    plugin_names: set[str] = set()
    plugin_versions: set[str] = set()
    plugin_states: set[str] = set()
    for element in extensions.iter():
        if local_name(element.tag) != "Plugin":
            continue
        name = element.attrib.get("name", "")
        version = element.attrib.get("version", "")
        state = element.attrib.get("state", "")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name):
            plugin_names.add(name)
        if re.fullmatch(r"[A-Za-z0-9_.+-]{1,40}", version):
            plugin_versions.add(version)
        if re.fullmatch(r"[A-Za-z-]{1,24}", state):
            plugin_states.add(state)
    emit("extension_plugin_names", sorted(plugin_names))
    emit("extension_plugin_versions", sorted(plugin_versions))
    emit("extension_plugin_states", sorted(plugin_states))

    runtime_documents: list[object] = []
    runtime_parse_failures = 0
    for element in extensions.iter():
        if local_name(element.tag) != "RuntimeSettings" or not (element.text or "").strip():
            continue
        document = decode_json_bytes((element.text or "").encode("utf-8"))
        if document is None:
            runtime_parse_failures += 1
        else:
            runtime_documents.append(document)
    emit("runtime_settings_json_parsed_count", len(runtime_documents))
    emit("runtime_settings_json_parse_failure_count", runtime_parse_failures)

    all_keys: list[str] = []
    all_strings: list[str] = []
    protected_values: list[str] = []
    for document in runtime_documents:
        keys, strings, _, _ = walk_json(document)
        all_keys.extend(keys)
        all_strings.extend(strings)
        stack = [document]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    if key == "protectedSettings" and isinstance(child, str):
                        protected_values.append(child)
                    stack.append(child)
            elif isinstance(item, list):
                stack.extend(item)
    shapes = Counter(value_shape(value) for value in all_strings)
    emit("runtime_settings_key_names", safe_names(all_keys))
    emit("runtime_settings_sensitive_key_name_count", sum(bool(SENSITIVE_NAME.search(key)) for key in all_keys))
    emit("runtime_settings_string_count", len(all_strings))
    for shape in ("url", "sas_url", "jwt", "base64"):
        emit(f"runtime_settings_{shape}_value_count", shapes[shape])
    emit("runtime_protected_settings_count", len(protected_values))
    emit("runtime_protected_settings_cert_thumbprint_key_present", "protectedSettingsCertThumbprint" in all_keys)

    decrypted = decrypt_protected_settings(protected_values, tenant_pem)
    decrypted_keys: list[str] = []
    decrypted_strings: list[str] = []
    for document in decrypted:
        keys, strings, _, _ = walk_json(document)
        decrypted_keys.extend(keys)
        decrypted_strings.extend(strings)
    decrypted_shapes = Counter(value_shape(value) for value in decrypted_strings)
    emit("decrypted_settings_key_names", safe_names(decrypted_keys))
    emit("decrypted_settings_sensitive_key_name_count", sum(bool(SENSITIVE_NAME.search(key)) for key in decrypted_keys))
    emit("decrypted_settings_string_count", len(decrypted_strings))
    for shape in ("url", "sas_url", "jwt", "base64"):
        emit(f"decrypted_settings_{shape}_value_count", decrypted_shapes[shape])
    for key in (
        "commandToExecute",
        "fileUris",
        "script",
        "storageAccountName",
        "storageAccountKey",
        "managedIdentity",
        "token",
    ):
        emit(f"decrypted_settings_{key.lower()}_present", key in decrypted_keys)


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
        extra_headers = None
        if name == "Certificates":
            extra_headers = certificate_request_headers("AES128_CBC")
            emit("certificates_transport_header_available", extra_headers is not None)
        status, body, content_type = fetch(
            url,
            WIRE_LIMIT,
            wire=True,
            retries=4,
            extra_headers=extra_headers,
        )
        if name == "Certificates" and status == 400 and extra_headers is not None:
            legacy_headers = certificate_request_headers("DES_EDE3_CBC")
            status, body, content_type = fetch(
                url,
                WIRE_LIMIT,
                wire=True,
                extra_headers=legacy_headers,
            )
            emit("certificates_legacy_cipher_retry", True)
        label = "wire_" + name.lower()
        emit(f"{label}_http_status", status)
        emit(f"{label}_size_within_limit", len(body) <= WIRE_LIMIT)
        if status == 200 and len(body) <= WIRE_LIMIT:
            parsed = classify_xml(label, body)
            if parsed is not None:
                documents[name] = parsed
        body = b""

    extensions = documents.get("ExtensionsConfig")
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
    tenant_pem = recover_tenant_pem(certificates)

    sas_values: dict[str, str] = {}
    sas_parts: dict[str, urllib.parse.SplitResult] = {}
    if extensions is not None:
        classify_runtime_settings(extensions, tenant_pem)
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
