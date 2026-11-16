"""Bounded passive T4 runner-boundary classification for Meta LHE.

The script performs no network request, opens no socket, and emits no raw
credential, URL, address, hostname, DNS suffix, identifier, file body, or
environment value. It reports only local schema names and aggregate classes.
"""

from __future__ import annotations

import base64
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
        emit("hca_jwt_algorithm", algorithm if algorithm in ("EdDSA",) else "other")
    if isinstance(payload, dict):
        import time

        now = int(time.time())
        for claim in ("iat", "rbf", "ret", "exp", "nbf"):
            emit("hca_jwt_" + claim + "_class", time_claim_class(payload.get(claim), now))
        emit("hca_jwt_audience_present", "aud" in payload)
        emit("hca_jwt_scope_present", "scope" in payload or "scp" in payload)
        emit("hca_jwt_signed_worker_id_present", isinstance(payload.get("wid"), str))

    binary = Path("/opt/hca/hosted-compute-agent")
    emit("hca_binary_present", binary.is_file())
    if binary.is_file():
        binary_metadata = binary.stat()
        emit("hca_binary_mode", stat.S_IMODE(binary_metadata.st_mode))
        emit("hca_binary_uid", binary_metadata.st_uid)
        emit("hca_binary_gid", binary_metadata.st_gid)


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
