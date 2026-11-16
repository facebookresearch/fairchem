"""Bounded passive T4 runner-boundary classification for Meta LHE.

The script performs no network request, opens no socket, and emits no raw
credential, URL, address, hostname, DNS suffix, identifier, file body, or
environment value. It reports only local schema names and aggregate classes.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
import re
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


MARKER = "META_LHE_FAIRCHEM_T4_PASSIVE"
MAX_TEXT_FILE = 2 * 1024 * 1024
MAX_COMMAND_OUTPUT = 4 * 1024 * 1024
KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
SCOPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,79}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
JWT_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"([A-Za-z0-9_-]{8,512}\.[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,4096})"
    r"(?![A-Za-z0-9_-])"
)
URL_TEXT_PATTERN = re.compile(r"https?://[^\s\"'<>]{1,4096}")
SECRET_NAME_PATTERN = re.compile(
    r"(?:auth|credential|key|password|sas|secret|token)", re.IGNORECASE
)
NETWORK_TERMS = {
    "network_configuration_id": re.compile(
        rb"network[_-]?configuration[_-]?id", re.IGNORECASE
    ),
    "vnet": re.compile(rb"\bvnet\b|virtual[_ -]?network", re.IGNORECASE),
    "subnet": re.compile(rb"\bsubnet\b", re.IGNORECASE),
    "private_dns": re.compile(rb"private[_ -]?dns|privatelink", re.IGNORECASE),
    "network_profile": re.compile(rb"network[_ -]?profile", re.IGNORECASE),
}


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    elif isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        rendered = str(value)
    print(f"{MARKER}_{name}={rendered}", flush=True)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def safe_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(
        key for key in value if isinstance(key, str) and KEY_PATTERN.fullmatch(key)
    )


def value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        if not value:
            return "empty_string"
        if UUID_PATTERN.fullmatch(value):
            return "uuid_string"
        if len(value.split(".")) == 3 and all(value.split(".")):
            return "jwt_shaped_string"
        parsed = urlsplit(value)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            return "url_string"
        if value.startswith("/"):
            return "absolute_path_string"
        return "opaque_string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "other"


def safe_scope_names(value: object) -> tuple[list[str], int]:
    if isinstance(value, str):
        candidates = re.split(r"[\s,]+", value.strip())
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return [], 0
    accepted: set[str] = set()
    rejected = 0
    for candidate in candidates:
        if not candidate:
            continue
        if (
            SCOPE_PATTERN.fullmatch(candidate)
            and not UUID_PATTERN.fullmatch(candidate)
            and not re.fullmatch(r"[0-9a-fA-F]{24,}", candidate)
        ):
            accepted.add(candidate)
        else:
            rejected += 1
    return sorted(accepted), rejected


def classify_claim_value(value: object) -> dict[str, object]:
    if isinstance(value, list):
        return {
            "kind": "list",
            "count": len(value),
            "item_types": sorted({value_type(item) for item in value}),
        }
    if isinstance(value, str):
        if UUID_PATTERN.fullmatch(value):
            return {"kind": "uuid"}
        parsed = urlsplit(value)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            return {"kind": "url", "url_class": classify_url(value)}
        if KEY_PATTERN.fullmatch(value) and len(value) <= 40:
            return {"kind": "short_identifier"}
        return {"kind": "opaque_string", "length_bucket": length_bucket(len(value))}
    return {"kind": value_type(value)}


def length_bucket(length: int) -> str:
    if length == 0:
        return "empty"
    if length <= 64:
        return "1_to_64"
    if length <= 512:
        return "65_to_512"
    if length <= 4096:
        return "513_to_4096"
    if length <= 65536:
        return "4097_to_65536"
    return "over_65536"


def classify_ip(value: object) -> str:
    if not isinstance(value, str):
        return "absent"
    value = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return "invalid"
    family = "ipv4" if address.version == 4 else "ipv6"
    if address.is_loopback:
        return family + "_loopback"
    if address.is_link_local:
        return family + "_link_local"
    if address.is_private:
        return family + "_private"
    if address.is_multicast:
        return family + "_multicast"
    if address.is_unspecified:
        return family + "_unspecified"
    if address.is_reserved:
        return family + "_reserved"
    return family + "_global"


def classify_network(value: object) -> str:
    if not isinstance(value, str) or value in ("default", "0.0.0.0/0", "::/0"):
        return "default"
    try:
        network = ipaddress.ip_network(value, strict=False)
    except ValueError:
        return "non_ip"
    address_class = classify_ip(str(network.network_address))
    return f"{address_class}_prefix_{network.prefixlen}"


def command_json(arguments: list[str]) -> object | None:
    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or len(result.stdout) > MAX_COMMAND_OUTPUT:
        return None
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def decode_jwt_part(value: str) -> object | None:
    try:
        raw = value.encode("ascii")
        raw += b"=" * ((4 - len(raw) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(raw))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None


def time_claim_class(value: object, now: int) -> str:
    if not isinstance(value, int):
        return "absent"
    delta = value - now
    if delta < -3600:
        return "past_over_1h"
    if delta < -60:
        return "past_1m_to_1h"
    if delta < 0:
        return "past_under_1m"
    if delta <= 60:
        return "future_under_1m"
    if delta <= 900:
        return "future_1m_to_15m"
    if delta <= 3600:
        return "future_15m_to_1h"
    return "future_over_1h"


def classify_url(value: object) -> dict[str, object]:
    result: dict[str, object] = {
        "present": isinstance(value, str) and bool(value),
        "https": False,
        "host_suffix_class": "absent",
        "path_segments": 0,
        "query_keys": [],
        "sas_permissions": "absent",
    }
    if not isinstance(value, str) or not value:
        return result
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    result["https"] = parsed.scheme == "https"
    if host.endswith(".github.net"):
        result["host_suffix_class"] = "github_net"
    elif host.endswith(".github.com"):
        result["host_suffix_class"] = "github_com"
    elif host.endswith(".actions.githubusercontent.com"):
        result["host_suffix_class"] = "github_actions_content"
    elif host.endswith(".visualstudio.com") or host == "dev.azure.com":
        result["host_suffix_class"] = "azure_devops"
    elif host.endswith(".blob.core.windows.net"):
        result["host_suffix_class"] = "azure_blob"
    elif host.endswith(".azure.com") or host.endswith(".azure.net"):
        result["host_suffix_class"] = "azure"
    elif host:
        result["host_suffix_class"] = "other"
    result["path_segments"] = len([part for part in parsed.path.split("/") if part])
    query = parse_qs(parsed.query, keep_blank_values=True)
    result["query_keys"] = sorted(
        key for key in query if KEY_PATTERN.fullmatch(key)
    )
    permissions = query.get("sp", [])
    if len(permissions) == 1 and re.fullmatch(r"[a-z]{1,20}", permissions[0]):
        result["sas_permissions"] = "".join(sorted(set(permissions[0])))
    return result


def inspect_hca() -> None:
    settings_path = Path("/opt/hca/.settings")
    emit("hca_settings_present", settings_path.is_file())
    if not settings_path.is_file():
        return
    metadata = settings_path.stat()
    emit("hca_settings_mode", stat.S_IMODE(metadata.st_mode))
    emit("hca_settings_uid", metadata.st_uid)
    emit("hca_settings_gid", metadata.st_gid)
    try:
        settings = json.loads(settings_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        emit("hca_settings_json", False)
        return
    emit("hca_settings_json", True)
    keys = safe_keys(settings)
    emit("hca_settings_keys", keys)
    lowered = {key.lower(): key for key in keys}
    for label, fragments in {
        "auth_token": ("authtoken", "auth_token"),
        "scheduler_url": ("schedulerapiurl", "scheduler_api_url"),
        "diagnostics_sas": ("diagnosticssasuri", "diagnostics_sas_uri"),
        "network_configuration": ("networkconfigurationid", "network_configuration_id"),
        "vnet": ("vnet", "virtualnetwork"),
        "subnet": ("subnet",),
    }.items():
        emit(
            "hca_key_" + label + "_present",
            any(fragment in lowered for fragment in fragments),
        )

    def get_setting(*names: str) -> object | None:
        for name in names:
            if name in settings:
                return settings[name]
        return None

    scheduler_url = get_setting("schedulerApiUrl", "SchedulerApiUrl")
    diagnostics_url = get_setting("diagnosticsSasUri", "DiagnosticsSasUri")
    emit("hca_scheduler_url_class", classify_url(scheduler_url))
    emit("hca_diagnostics_url_class", classify_url(diagnostics_url))

    mode = get_setting("mode", "Mode")
    emit(
        "hca_mode_class",
        mode if mode in ("requestorchestrator", "runner") else "other",
    )
    properties = get_setting("properties", "Properties")
    emit("hca_properties_present", isinstance(properties, dict))
    emit("hca_property_keys", safe_keys(properties))
    if isinstance(properties, dict):
        emit(
            "hca_property_value_types",
            {
                key: value_type(properties[key])
                for key in safe_keys(properties)
            },
        )
        emit(
            "hca_properties_equal_auth_token_keys",
            sorted(
                key
                for key in safe_keys(properties)
                if isinstance(get_setting("authToken", "AuthToken"), str)
                and properties[key] == get_setting("authToken", "AuthToken")
            ),
        )
        emit(
            "hca_properties_equal_scheduler_url_keys",
            sorted(
                key
                for key in safe_keys(properties)
                if isinstance(scheduler_url, str) and properties[key] == scheduler_url
            ),
        )

    runner_settings = get_setting("runnerSettings", "RunnerSettings")
    emit("hca_runner_settings_present", isinstance(runner_settings, dict))
    if isinstance(runner_settings, dict):
        emit("hca_runner_settings_entry_count", len(runner_settings))
        nested_keys: set[str] = set()
        nested_types: Counter[str] = Counter()
        for entry in runner_settings.values():
            if not isinstance(entry, dict):
                nested_types[value_type(entry)] += 1
                continue
            for key in safe_keys(entry):
                nested_keys.add(key)
                nested_types[f"{key}:{value_type(entry[key])}"] += 1
        emit("hca_runner_settings_field_names", sorted(nested_keys))
        emit("hca_runner_settings_field_types", dict(sorted(nested_types.items())))

    for field_name, setting_names in {
        "runner_command": ("runnerCommand", "RunnerCommand"),
        "runner_working_directory": (
            "runnerAgentWorkingDirectory",
            "RunnerAgentWorkingDirectory",
        ),
        "vm_name": ("vmName", "VMName"),
        "config_id": ("configId", "ConfigId"),
        "quota_id": ("quotaId", "QuotaId"),
        "mac_address": ("macAddress", "MacAddress"),
        "trace_url": ("traceApiUrl", "TraceApiUrl"),
        "watchdog_trace_url": ("watchdogTraceApiUrl", "WatchdogTraceApiUrl"),
    }.items():
        field_value = get_setting(*setting_names)
        emit("hca_field_" + field_name + "_type", value_type(field_value))
        if field_name.endswith("url"):
            emit("hca_field_" + field_name + "_class", classify_url(field_value))

    token = get_setting("authToken", "AuthToken")
    if not isinstance(token, str):
        emit("hca_jwt_present", False)
        return
    pieces = token.split(".")
    emit("hca_jwt_present", len(pieces) == 3 and all(pieces))
    if len(pieces) != 3 or not all(pieces):
        return
    header = decode_jwt_part(pieces[0])
    payload = decode_jwt_part(pieces[1])
    emit("hca_jwt_header_keys", safe_keys(header))
    emit("hca_jwt_claim_names", safe_keys(payload))
    if isinstance(header, dict):
        algorithm = header.get("alg")
        emit(
            "hca_jwt_algorithm",
            algorithm if algorithm in ("EdDSA", "RS256", "ES256") else "other",
        )
    if isinstance(payload, dict):
        import time

        now = int(time.time())
        for claim in ("iat", "rbf", "ret", "exp", "nbf"):
            emit("hca_jwt_" + claim + "_class", time_claim_class(payload.get(claim), now))
        emit("hca_jwt_audience_present", "aud" in payload)
        emit("hca_jwt_scope_present", "scope" in payload or "scp" in payload)
        emit("hca_jwt_signed_worker_id_present", isinstance(payload.get("wid"), str))
        emit("hca_jwt_audience_class", classify_claim_value(payload.get("aud")))
        emit("hca_jwt_issuer_class", classify_claim_value(payload.get("iss")))
        emit(
            "hca_jwt_identity_type_class",
            classify_claim_value(payload.get("IdentityTypeClaim")),
        )
        scope_value = payload.get("scp", payload.get("scope"))
        scope_names, rejected_scope_count = safe_scope_names(scope_value)
        emit("hca_jwt_scope_names", scope_names)
        emit("hca_jwt_unclassified_scope_count", rejected_scope_count)

    binary = Path("/opt/hca/hosted-compute-agent")
    emit("hca_binary_present", binary.is_file())
    if binary.is_file():
        binary_metadata = binary.stat()
        emit("hca_binary_mode", stat.S_IMODE(binary_metadata.st_mode))
        emit("hca_binary_uid", binary_metadata.st_uid)
        emit("hca_binary_gid", binary_metadata.st_gid)

    inspect_hca_file_and_process_boundary(binary)


def inspect_hca_file_and_process_boundary(binary: Path) -> None:
    root = Path("/opt/hca")
    regular_files = 0
    root_owned_files = 0
    world_writable_files = 0
    world_writable_executables = 0
    symlinks = 0
    if root.is_dir():
        for path in root.rglob("*"):
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                symlinks += 1
            elif stat.S_ISREG(metadata.st_mode):
                regular_files += 1
                if metadata.st_uid == 0:
                    root_owned_files += 1
                if metadata.st_mode & stat.S_IWOTH:
                    world_writable_files += 1
                    if metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                        world_writable_executables += 1
    emit("hca_regular_file_count", regular_files)
    emit("hca_root_owned_file_count", root_owned_files)
    emit("hca_world_writable_file_count", world_writable_files)
    emit("hca_world_writable_executable_count", world_writable_executables)
    emit("hca_symlink_count", symlinks)
    emit("hca_binary_writable_by_job", binary.is_file() and os.access(binary, os.W_OK))

    process_count = 0
    root_process_count = 0
    disk_inode_match_count = 0
    deleted_mapping_count = 0
    command_argument_counts: Counter[str] = Counter()
    try:
        binary_metadata = binary.stat()
    except OSError:
        binary_metadata = None
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            executable = (process_root / "exe").resolve(strict=True)
        except OSError:
            continue
        if executable != binary:
            continue
        process_count += 1
        try:
            status = (process_root / "status").read_text(errors="replace")
        except OSError:
            status = ""
        uid_match = re.search(r"^Uid:\s+(\d+)", status, re.MULTILINE)
        if uid_match and uid_match.group(1) == "0":
            root_process_count += 1
        try:
            process_metadata = (process_root / "exe").stat()
        except OSError:
            process_metadata = None
        if (
            binary_metadata is not None
            and process_metadata is not None
            and (binary_metadata.st_dev, binary_metadata.st_ino)
            == (process_metadata.st_dev, process_metadata.st_ino)
        ):
            disk_inode_match_count += 1
        try:
            maps = (process_root / "maps").read_text(errors="replace")
        except OSError:
            maps = ""
        if "(deleted)" in maps:
            deleted_mapping_count += 1
        try:
            arguments = (process_root / "cmdline").read_bytes().split(b"\0")
        except OSError:
            arguments = []
        argument_count = sum(bool(argument) for argument in arguments)
        command_argument_counts[length_bucket(argument_count)] += 1
    emit("hca_running_process_count", process_count)
    emit("hca_root_process_count", root_process_count)
    emit("hca_process_disk_inode_match_count", disk_inode_match_count)
    emit("hca_process_deleted_mapping_count", deleted_mapping_count)
    emit("hca_process_argument_count_buckets", dict(sorted(command_argument_counts.items())))

    service_units = 0
    service_units_root_owned = 0
    service_units_world_writable = 0
    restart_policies: Counter[str] = Counter()
    for unit_root in (
        Path("/etc/systemd/system"),
        Path("/lib/systemd/system"),
        Path("/usr/lib/systemd/system"),
    ):
        if not unit_root.is_dir():
            continue
        for unit in unit_root.rglob("*.service"):
            try:
                data = unit.read_bytes()
                metadata = unit.stat()
            except OSError:
                continue
            if len(data) > 1024 * 1024 or b"hosted-compute-agent" not in data:
                continue
            service_units += 1
            if metadata.st_uid == 0:
                service_units_root_owned += 1
            if metadata.st_mode & stat.S_IWOTH:
                service_units_world_writable += 1
            text = data.decode("utf-8", "replace")
            for policy in re.findall(r"(?mi)^Restart\s*=\s*([A-Za-z-]+)\s*$", text):
                if policy in (
                    "no",
                    "always",
                    "on-success",
                    "on-failure",
                    "on-abnormal",
                    "on-abort",
                    "on-watchdog",
                ):
                    restart_policies[policy] += 1
                else:
                    restart_policies["other"] += 1
    emit("hca_systemd_service_unit_count", service_units)
    emit("hca_systemd_service_root_owned_count", service_units_root_owned)
    emit("hca_systemd_service_world_writable_count", service_units_world_writable)
    emit("hca_systemd_restart_policies", dict(sorted(restart_policies.items())))


def inspect_addresses_and_routes() -> None:
    addresses = command_json(["ip", "-json", "address", "show"])
    address_classes: Counter[str] = Counter()
    prefix_classes: Counter[str] = Counter()
    interface_count = 0
    non_loopback_interface_count = 0
    if isinstance(addresses, list):
        interface_count = len(addresses)
        for interface in addresses:
            if not isinstance(interface, dict):
                continue
            if interface.get("ifname") != "lo":
                non_loopback_interface_count += 1
            for record in interface.get("addr_info", []):
                if not isinstance(record, dict):
                    continue
                address_class = classify_ip(record.get("local"))
                address_classes[address_class] += 1
                prefix = record.get("prefixlen")
                if isinstance(prefix, int):
                    prefix_classes[f"{address_class}_prefix_{prefix}"] += 1
    emit("network_interface_count", interface_count)
    emit("network_non_loopback_interface_count", non_loopback_interface_count)
    emit("network_address_classes", dict(sorted(address_classes.items())))
    emit("network_address_prefix_classes", dict(sorted(prefix_classes.items())))

    route_classes: Counter[str] = Counter()
    route_protocols: Counter[str] = Counter()
    for family in ("-4", "-6"):
        routes = command_json(["ip", family, "-json", "route", "show", "table", "all"])
        if not isinstance(routes, list):
            continue
        for route in routes:
            if not isinstance(route, dict):
                continue
            destination = classify_network(route.get("dst", "default"))
            gateway = classify_ip(route.get("gateway"))
            route_type = route.get("type", "unicast")
            if not isinstance(route_type, str) or not KEY_PATTERN.fullmatch(route_type):
                route_type = "other"
            route_classes[f"{family[1:]}_{destination}_via_{gateway}_{route_type}"] += 1
            protocol = route.get("protocol", "absent")
            if isinstance(protocol, str) and KEY_PATTERN.fullmatch(protocol):
                route_protocols[protocol] += 1
            else:
                route_protocols["other"] += 1
    emit("network_route_classes", dict(sorted(route_classes.items())))
    emit("network_route_protocols", dict(sorted(route_protocols.items())))


def suffix_flags(names: list[str]) -> dict[str, bool]:
    lowered = [name.lower().rstrip(".") for name in names]
    return {
        "azure_internal": any(name.endswith(".internal.cloudapp.net") for name in lowered),
        "generic_internal": any(
            name.endswith(suffix)
            for name in lowered
            for suffix in (".internal", ".corp", ".local")
        ),
        "privatelink": any("privatelink" in name for name in lowered),
        "meta_internal": any(
            name.endswith(suffix)
            for name in lowered
            for suffix in (".tfbnw.net", ".facebook.com", ".meta.com")
        ),
    }


def inspect_resolver_and_hosts() -> None:
    resolver = Path("/etc/resolv.conf")
    nameserver_classes: Counter[str] = Counter()
    search_names: list[str] = []
    if resolver.is_file():
        try:
            text = resolver.read_text(errors="replace")[:262144]
        except OSError:
            text = ""
        for line in text.splitlines():
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "nameserver" and len(fields) >= 2:
                nameserver_classes[classify_ip(fields[1])] += 1
            elif fields[0] in ("search", "domain"):
                search_names.extend(fields[1:])
    emit("resolver_nameserver_classes", dict(sorted(nameserver_classes.items())))
    emit("resolver_search_suffix_count", len(search_names))
    emit("resolver_search_suffix_flags", suffix_flags(search_names))

    hosts = Path("/etc/hosts")
    host_address_classes: Counter[str] = Counter()
    host_aliases: list[str] = []
    if hosts.is_file():
        try:
            text = hosts.read_text(errors="replace")[:262144]
        except OSError:
            text = ""
        for line in text.splitlines():
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                host_address_classes[classify_ip(fields[0])] += 1
                host_aliases.extend(fields[1:])
    emit("hosts_address_classes", dict(sorted(host_address_classes.items())))
    emit("hosts_alias_count", len(host_aliases))
    emit("hosts_suffix_flags", suffix_flags(host_aliases))


def decode_custom_data(value: str) -> tuple[bytes, str]:
    compact = re.sub(r"\s+", "", value)
    try:
        encoded = compact.encode("ascii")
    except UnicodeEncodeError:
        return value.encode("utf-8", "replace"), "plain_text"
    if len(encoded) > MAX_TEXT_FILE * 2:
        return b"", "oversized"
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return value.encode("utf-8", "replace"), "plain_text"
    if len(decoded) > MAX_TEXT_FILE:
        return b"", "oversized_base64"
    return decoded, "base64"


def text_format(data: bytes) -> str:
    if not data:
        return "empty"
    if b"\x00" in data[:4096]:
        return "binary"
    text = data.decode("utf-8", "replace").lstrip()
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    if replacement_ratio > 0.01:
        return "binary"
    if text.startswith("#!"):
        return "shell"
    if text.startswith("#cloud-config"):
        return "cloud_config"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return "json"
    try:
        ET.fromstring(text)
    except ET.ParseError:
        pass
    else:
        return "xml"
    if re.search(r"(?m)^[A-Za-z][A-Za-z0-9_.-]{0,79}:\s*", text):
        return "yaml_like"
    return "text"


def collect_json_keys(value: object, destination: set[str], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and KEY_PATTERN.fullmatch(key):
                destination.add(key)
            collect_json_keys(child, destination, depth + 1)
    elif isinstance(value, list):
        for child in value[:1024]:
            collect_json_keys(child, destination, depth + 1)


def inspect_custom_data_values(values: list[str]) -> None:
    emit("waagent_custom_data_element_count", len(values))
    if not values:
        return
    hca_settings: dict[str, object] = {}
    try:
        loaded_settings = json.loads(Path("/opt/hca/.settings").read_bytes())
    except (OSError, json.JSONDecodeError):
        loaded_settings = None
    if isinstance(loaded_settings, dict):
        hca_settings = loaded_settings
    auth_token = hca_settings.get("authToken", hca_settings.get("AuthToken"))
    scheduler_url = hca_settings.get(
        "schedulerApiUrl", hca_settings.get("SchedulerApiUrl")
    )

    encodings: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    encoded_length_buckets: Counter[str] = Counter()
    decoded_length_buckets: Counter[str] = Counter()
    key_names: set[str] = set()
    secret_key_names: set[str] = set()
    url_classes: Counter[str] = Counter()
    jwt_claim_names: set[str] = set()
    jwt_scope_names: set[str] = set()
    jwt_unclassified_scopes = 0
    jwt_count = 0
    jwt_matches_hca_auth = 0
    contains_hca_auth = False
    contains_scheduler_url = False
    private_key_marker_count = 0
    command_markers: Counter[str] = Counter()

    for value in values:
        encoded_length_buckets[length_bucket(len(value))] += 1
        data, encoding = decode_custom_data(value)
        encodings[encoding] += 1
        decoded_length_buckets[length_bucket(len(data))] += 1
        format_name = text_format(data)
        formats[format_name] += 1
        if format_name == "binary" or not data:
            continue
        text = data.decode("utf-8", "replace")
        if isinstance(auth_token, str) and auth_token:
            contains_hca_auth = contains_hca_auth or auth_token in text
        if isinstance(scheduler_url, str) and scheduler_url:
            contains_scheduler_url = contains_scheduler_url or scheduler_url in text

        for key in re.findall(
            r"(?m)^\s*(?:export\s+)?([A-Za-z][A-Za-z0-9_.-]{0,79})\s*(?:=|:)" ,
            text,
        ):
            if KEY_PATTERN.fullmatch(key):
                key_names.add(key)
                if SECRET_NAME_PATTERN.search(key):
                    secret_key_names.add(key)
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = None
        collect_json_keys(parsed_json, key_names)
        if isinstance(parsed_json, (dict, list)):
            all_json_keys: set[str] = set()
            collect_json_keys(parsed_json, all_json_keys)
            secret_key_names.update(
                key for key in all_json_keys if SECRET_NAME_PATTERN.search(key)
            )
        try:
            parsed_xml = ET.fromstring(text)
        except ET.ParseError:
            parsed_xml = None
        if parsed_xml is not None:
            for element in parsed_xml.iter():
                name = local_name(element.tag)
                if KEY_PATTERN.fullmatch(name):
                    key_names.add(name)
                    if SECRET_NAME_PATTERN.search(name):
                        secret_key_names.add(name)

        for raw_url in URL_TEXT_PATTERN.findall(text):
            classification = classify_url(raw_url.rstrip(".,);]"))
            url_key = (
                str(classification["host_suffix_class"])
                + "_https_"
                + ("yes" if classification["https"] else "no")
                + "_segments_"
                + str(classification["path_segments"])
            )
            url_classes[url_key] += 1

        for match in JWT_TEXT_PATTERN.finditer(text):
            candidate = match.group(1)
            jwt_count += 1
            if isinstance(auth_token, str) and candidate == auth_token:
                jwt_matches_hca_auth += 1
            pieces = candidate.split(".")
            payload = decode_jwt_part(pieces[1])
            if not isinstance(payload, dict):
                continue
            jwt_claim_names.update(safe_keys(payload))
            scopes, rejected = safe_scope_names(
                payload.get("scp", payload.get("scope"))
            )
            jwt_scope_names.update(scopes)
            jwt_unclassified_scopes += rejected

        private_key_marker_count += len(
            re.findall(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", text)
        )
        for marker, pattern in {
            "hosted_compute_agent": r"hosted-compute-agent",
            "actions_runner": r"(?:actions-runner|Runner\.Listener)",
            "systemd": r"\bsystemctl\b|\.service\b",
            "cloud_init": r"\bcloud-init\b|#cloud-config",
            "metadata_service": r"169\.254\.169\.254|metadata/instance",
        }.items():
            if re.search(pattern, text, re.IGNORECASE):
                command_markers[marker] += 1

    emit("waagent_custom_data_encodings", dict(sorted(encodings.items())))
    emit("waagent_custom_data_formats", dict(sorted(formats.items())))
    emit(
        "waagent_custom_data_encoded_length_buckets",
        dict(sorted(encoded_length_buckets.items())),
    )
    emit(
        "waagent_custom_data_decoded_length_buckets",
        dict(sorted(decoded_length_buckets.items())),
    )
    emit("waagent_custom_data_key_names", sorted(key_names))
    emit("waagent_custom_data_secret_key_names", sorted(secret_key_names))
    emit("waagent_custom_data_url_classes", dict(sorted(url_classes.items())))
    emit("waagent_custom_data_jwt_count", jwt_count)
    emit("waagent_custom_data_jwt_claim_names", sorted(jwt_claim_names))
    emit("waagent_custom_data_jwt_scope_names", sorted(jwt_scope_names))
    emit(
        "waagent_custom_data_jwt_unclassified_scope_count",
        jwt_unclassified_scopes,
    )
    emit("waagent_custom_data_jwt_matches_hca_auth_count", jwt_matches_hca_auth)
    emit("waagent_custom_data_contains_hca_auth", contains_hca_auth)
    emit("waagent_custom_data_contains_scheduler_url", contains_scheduler_url)
    emit("waagent_custom_data_private_key_marker_count", private_key_marker_count)
    emit("waagent_custom_data_command_markers", dict(sorted(command_markers.items())))


def inspect_waagent() -> None:
    root = Path("/var/lib/waagent")
    emit("waagent_directory_present", root.is_dir())
    if not root.is_dir():
        return
    try:
        files = [path for path in root.iterdir() if path.is_file()]
    except OSError:
        files = []
    emit("waagent_top_level_file_count", len(files))
    emit("waagent_goalstate_file_count", sum("GoalState" in path.name for path in files))
    emit(
        "waagent_extensions_config_file_count",
        sum("ExtensionsConfig" in path.name for path in files),
    )
    emit("waagent_ovf_environment_present", (root / "ovf-env.xml").is_file())

    xml_tag_counts: Counter[str] = Counter()
    url_classes: Counter[str] = Counter()
    custom_data_nonempty = False
    custom_data_values: list[str] = []
    password_element_count = 0
    ssh_public_key_element_count = 0
    candidate_files = sorted(
        [
            path
            for path in files
            if path.name == "ovf-env.xml"
            or "GoalState" in path.name
            or "ExtensionsConfig" in path.name
        ],
        key=lambda path: path.stat().st_mtime_ns,
    )[-12:]
    for path in candidate_files:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if len(data) > MAX_TEXT_FILE:
            continue
        try:
            document = ET.fromstring(data)
        except ET.ParseError:
            continue
        for element in document.iter():
            name = local_name(element.tag)
            if KEY_PATTERN.fullmatch(name):
                xml_tag_counts[name] += 1
            text = (element.text or "").strip()
            if name.lower() == "customdata" and text:
                custom_data_nonempty = True
                custom_data_values.append(text)
            if name.lower() == "password":
                password_element_count += 1
            if name.lower() in ("fingerprint", "value") and "ssh" in path.name.lower():
                ssh_public_key_element_count += 1
            if text.startswith(("https://", "http://")):
                classification = classify_url(text)
                key = (
                    str(classification["host_suffix_class"])
                    + "_https_"
                    + ("yes" if classification["https"] else "no")
                    + "_sas_"
                    + str(classification["sas_permissions"])
                )
                url_classes[key] += 1
    emit("waagent_xml_tag_names", sorted(xml_tag_counts))
    emit("waagent_xml_url_classes", dict(sorted(url_classes.items())))
    emit("waagent_custom_data_nonempty", custom_data_nonempty)
    emit("waagent_password_element_count", password_element_count)
    emit("waagent_ssh_key_like_element_count", ssh_public_key_element_count)
    inspect_custom_data_values(custom_data_values)


def inspect_local_network_configuration_markers() -> None:
    roots = (Path("/opt/hca"), Path("/var/lib/waagent"), Path("/run/cloud-init"))
    hits: Counter[str] = Counter()
    scanned_files = 0
    skipped_large_or_binary = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_TEXT_FILE:
                skipped_large_or_binary += 1
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                skipped_large_or_binary += 1
                continue
            scanned_files += 1
            for label, pattern in NETWORK_TERMS.items():
                if pattern.search(data):
                    hits[label] += 1
    emit("local_config_text_files_scanned", scanned_files)
    emit("local_config_large_or_binary_files_skipped", skipped_large_or_binary)
    emit("local_network_marker_file_counts", dict(sorted(hits.items())))


def inspect_environment_names() -> None:
    names = sorted(
        name
        for name in os.environ
        if KEY_PATTERN.fullmatch(name)
        and re.search(r"AZURE|AWS|GOOGLE|GCP|VNET|SUBNET|NETWORK|OIDC", name, re.I)
    )
    emit("cloud_or_network_environment_names", names)
    emit("actions_oidc_request_name_present", any("OIDC" in name for name in names))


def main() -> int:
    emit("schema", "v1")
    emit("effective_uid_zero", os.geteuid() == 0)
    inspect_hca()
    inspect_addresses_and_routes()
    inspect_resolver_and_hosts()
    inspect_waagent()
    inspect_local_network_configuration_markers()
    inspect_environment_names()
    emit("network_requests", 0)
    emit("sockets_opened", 0)
    emit("raw_addresses_hostnames_urls_or_identifiers_emitted", 0)
    emit("raw_credentials_or_file_bodies_emitted", 0)
    emit("state_changes", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
