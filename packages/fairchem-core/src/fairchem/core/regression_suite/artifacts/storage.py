"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# ``MeasurementRecord.values`` is a NumPy array, not a pandas object.
# ruff: noqa: PD011
import dataclasses
import json
import os
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core.regression_suite.spec.hashing import file_sha256, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    REFERENCE_SCHEMA,
    SCHEMA_VERSION,
    CommandError,
    MeasurementRecord,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """
    Write a JSON document atomically in its destination directory.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary, payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def staged_directory(
    destination: str | Path, *, replace: bool = False
) -> Iterator[Path]:
    """
    Yield a sibling directory that becomes ``destination`` if the block succeeds.

    Nothing is published when the block fails. An existing destination is
    replaced only with ``replace``, and restored if the swap fails. Unlike
    ``tempfile`` directories, the result gets the usual umask permissions.
    """
    destination = Path(destination)
    if destination.exists() and not replace:
        raise FileExistsError(f"already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
    previous = temporary.with_name(f"{temporary.name}.previous")
    temporary.mkdir()
    try:
        yield temporary
        if replace and destination.exists():
            os.rename(destination, previous)
        try:
            os.rename(temporary, destination)
        except BaseException:
            if previous.exists():
                os.rename(previous, destination)
            raise
    finally:
        for path in (temporary, previous):
            if path.exists():
                shutil.rmtree(path)


ArtifactPath = Path | zipfile.Path
BASELINE_SUFFIX = ".baseline"


def sealed_baseline_path(artifact_dir: str | Path) -> Path:
    """
    Return the sealed file that sits beside a baseline's run directory.
    """
    directory = Path(artifact_dir)
    return directory.with_name(directory.name + BASELINE_SUFFIX)


def baseline_run_directory(path: str | Path) -> Path:
    """
    Return the run directory of a sealed baseline file or run directory.
    """
    source = Path(path)
    return source.with_name(source.name.removesuffix(BASELINE_SUFFIX))


def open_artifact(path: str | Path) -> ArtifactPath:
    """
    Return the root of a run directory or of a sealed baseline file.

    Both expose the same ``baseline.json`` and ``fragments/`` layout, so
    readers never need to know which one they were given.
    """
    source = Path(path).expanduser().resolve()
    return zipfile.Path(source) if source.is_file() else source


def fragment_path(
    root: str | ArtifactPath, suite: str, profile: str, role: str
) -> ArtifactPath:
    """
    Return where the records of one suite, profile, and role live.
    """
    base = (Path(root) if isinstance(root, str) else root) / "fragments"
    base = base / suite / profile
    return base / "reference" if role == "reference" else base / role / "records"


def load_json(path: str | ArtifactPath) -> dict[str, Any]:
    """
    Load a JSON object and reject non-object documents.
    """
    source = Path(path) if isinstance(path, str) else path
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RegressionSchemaError(f"{path} must contain a JSON object")
    return payload


def load_sealed(path: str | ArtifactPath, key: str, label: str) -> dict[str, Any]:
    """
    Load a JSON document whose ``key`` must hash all of its other fields.
    """
    payload = load_json(path)
    unhashed = {name: value for name, value in payload.items() if name != key}
    if payload.get(key) != stable_hash(unhashed):
        raise RegressionSchemaError(f"{label} hash is invalid")
    return payload


_RECORD_FIELDS = tuple(
    field.name
    for field in dataclasses.fields(MeasurementRecord)
    if field.name != "values"
)


def _record_metadata(record: MeasurementRecord, array_key: str) -> dict[str, Any]:
    return {
        "array_key": array_key,
        "shape": list(record.values.shape),
        "dtype": "float64",
        **{field: getattr(record, field) for field in _RECORD_FIELDS},
    }


def write_reference(
    directory: str | Path,
    records: Sequence[MeasurementRecord],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Atomically write one run's full-resolution records and their manifest.

    FP64 references and every comparison run use this format. The manifest is
    written last. A directory without ``manifest.json`` is therefore incomplete
    and must not be consumed.
    """
    if not records:
        raise RegressionSchemaError("a reference requires at least one measurement")
    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise RegressionSchemaError("reference measurement keys must be unique")
    ordered = sorted(records, key=lambda record: record.key)

    with staged_directory(directory) as temporary:
        arrays = {
            f"measurement_{index:06d}": record.values
            for index, record in enumerate(ordered)
        }
        values_path = temporary / "values.npz"
        np.savez_compressed(values_path, **arrays)
        manifest = {
            "schema": REFERENCE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "identity": dict(identity),
            "values_file": "values.npz",
            "values_sha256": file_sha256(values_path),
            "record_count": len(ordered),
            "records": [
                _record_metadata(record, f"measurement_{index:06d}")
                for index, record in enumerate(ordered)
            ],
        }
        manifest["reference_hash"] = stable_hash(manifest)
        write_json(temporary / "manifest.json", manifest)
    return manifest


def load_reference(
    directory: str | ArtifactPath,
) -> tuple[dict[str, Any], list[MeasurementRecord]]:
    """
    Load a verified reference artifact and its full-resolution records.
    """
    source = Path(directory) if isinstance(directory, str) else directory
    manifest = load_reference_manifest(source)
    with (
        (source / str(manifest["values_file"])).open("rb") as handle,
        np.load(handle, allow_pickle=False) as arrays,
    ):
        records = [
            MeasurementRecord(
                values=arrays[item["array_key"]].copy(),
                **{field: item[field] for field in _RECORD_FIELDS},
            )
            for item in manifest["records"]
        ]
    return manifest, records


def load_reference_manifest(directory: str | ArtifactPath) -> dict[str, Any]:
    """
    Load a reference manifest after checking its hash and its values file.
    """
    source = Path(directory) if isinstance(directory, str) else directory
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise RegressionSchemaError(f"reference is incomplete: {source}")
    manifest = load_sealed(manifest_path, "reference_hash", "reference manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise RegressionSchemaError("reference schema version is unsupported")
    values_path = source / str(manifest["values_file"])
    if file_sha256(values_path) != manifest["values_sha256"]:
        raise RegressionSchemaError("reference values file hash is invalid")
    return manifest


def pack_baseline(artifact_dir: str | Path, destination: str | Path) -> Path:
    """
    Seal a finalized baseline and every run it references into one file.

    The file is a zip archive with the run directory's own layout:
    ``baseline.json`` plus the reference, calibration, and verification
    fragments and their performance sidecars. ``baseline_id`` hashes the
    fragment manifests, which hash their values, so the file is tamper-evident.
    """
    root = Path(artifact_dir).resolve()
    payload = load_json(root / "baseline.json")
    members = ["baseline.json"]
    for entry in payload["suites"].values():
        reference = entry["reference"]
        members += [reference["manifest"], reference["values"]]
        for collection in ("profiles", "verifications"):
            for run in entry[collection].values():
                records = Path(run["fragment"]).parent
                members += [str(records / "manifest.json"), str(records / "values.npz")]
                sidecar = records.parent / "performance.json"
                if (root / sidecar).is_file():
                    members.append(str(sidecar))
    output = Path(destination)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        # LZMA keeps every member byte-identical, so the hashes recorded in
        # the manifests still verify, and it shrinks a full baseline by a fifth.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_LZMA) as archive:
            for member in members:
                archive.write(root / member, member)
        sealed = zipfile.Path(temporary)
        for entry in payload["suites"].values():
            load_reference_manifest((sealed / entry["reference"]["manifest"]).parent)
            for collection in ("profiles", "verifications"):
                for run in entry[collection].values():
                    manifest = sealed / run["fragment"]
                    if file_sha256(manifest) != run["fragment_sha256"]:
                        raise RegressionSchemaError(f"{run['fragment']} changed")
                    load_reference_manifest(manifest.parent)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def checkout_root() -> Path:
    """
    Return the git checkout the suite is imported from, whatever the process
    working directory.
    """
    repository = repository_root()
    if not (repository / ".git").exists():
        raise CommandError(
            "the regression suite keeps its artifacts in regression-artifacts/ "
            "of its git checkout; run it from a fairchem checkout"
        )
    return repository


def artifact_root() -> Path:
    """
    Return ``regression-artifacts/`` in the checkout the suite is imported from.
    """
    return checkout_root() / "regression-artifacts"


def report_directory(root: str | Path, profile: str, mode: str, run: str) -> Path:
    """
    Return where the reports of one quick, fast, or full run are written.

    Every run's ``report.json``, ``index.html``, and ``wandb.json`` share one
    tree under ``root``, so results are found in one place; the raw run data
    they are built from stays in the run's own directory.

    Args:
        root: The artifact root.
        profile: The candidate profile as given on the command line.
        mode: ``quick``, ``fast``, or ``full``.
        run: What keeps runs of one profile and mode apart.
    """
    return Path(root) / "reports" / profile / mode / run
