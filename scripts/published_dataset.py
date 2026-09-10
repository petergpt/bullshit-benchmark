#!/usr/bin/env python3
"""Lossless, size-bounded storage for the four published benchmark assets.

``pack`` stages and verifies immutable, content-addressed parts, then atomically
activates their manifest. Existing parts are retained so an interrupted update
and readers holding the preceding manifest remain safe. Pruning is a separate
offline operation. Collection/judge inputs are never edited.

JSONL part concatenation is byte-for-byte identical to its input. Gzip parts are
standalone JSON arrays: join their nonempty array interiors with commas inside
``[]`` to recover the exact original UTF-8 JSON, without reserializing numbers.
The array must have no whitespace outside its brackets (the publisher already
writes this form). Checksums describe uncompressed logical bytes and encoded
part bytes separately. Detail arrays are grouped by question and sorted by
sample ID; their row contents are preserved while their logical order changes.
``cat`` reconstructs either format on standard output. Version 1 and legacy
single-file datasets remain readable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Iterator
import zlib


MAX_FILE_BYTES = 4 * 1024 * 1024
STORAGE_VERSION = 2
ASSET_FORMATS = {
    "responses.jsonl": "jsonl",
    "aggregate.jsonl": "jsonl",
    "viewer_rows.json.gz": "json-gzip",
    "viewer_details.json.gz": "json-gzip",
}
SOURCE_KEYS = {
    "responses.jsonl": "responses_file",
    "aggregate.jsonl": "aggregate_rows_file",
    "viewer_rows.json.gz": "viewer_rows_file",
    "viewer_details.json.gz": "viewer_details_file",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class StorageError(ValueError):
    """A declared dataset is incomplete, corrupt, or unsafe to read/write."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gzip_bytes(raw: bytes) -> bytes:
    # GzipFile fixes OS=255 across Python versions; gzip.compress(mtime=0)
    # used the platform's zlib OS header in some earlier Python releases.
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=output, mtime=0) as stream:
        stream.write(raw)
    return output.getvalue()


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StorageError(f"{label} must be an integer >= {minimum}")
    return value


def _hash(value, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise StorageError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _part_path(logical: str, index: int) -> str:
    """Version 1 path, retained only for reading earlier manifests."""
    stem, suffix = logical.split(".", 1)
    return f"{stem}/part-{index:05d}.{suffix}"


def _content_path(logical: str, digest: str) -> str:
    stem, suffix = logical.split(".", 1)
    return f"{stem}/sha256-{digest}.{suffix}"


def _safe_path(root: Path, relative: str) -> Path:
    # Storage paths are further restricted to the exact generated names below.
    # Check every directory component before any read, replacement, or cleanup.
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
        or "%" in relative
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or Path(relative).is_absolute()
    ):
        raise StorageError(f"Unsafe dataset-relative path: {relative!r}")
    path = root
    if path.is_symlink():
        raise StorageError(f"Symlink dataset directory is not allowed: {path}")
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise StorageError(f"Symlink asset path is not allowed: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise StorageError(f"Asset escapes dataset directory: {relative!r}")
    return path


def _file_bytes(root: Path, relative: str, part=None, maximum=None) -> bytes:
    path = _safe_path(root, relative)
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise StorageError(f"Missing dataset file: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise StorageError(f"Dataset asset is not a regular file: {path}")
    if maximum is not None and info.st_size > maximum:
        raise StorageError(f"Part exceeds max_file_bytes: {path}")
    if part is not None and info.st_size != part["bytes"]:
        raise StorageError(f"Part byte count mismatch: {path}")
    raw = path.read_bytes()
    if part is not None and (
        len(raw) != part["bytes"] or _sha256(raw) != part["sha256"]
    ):
        raise StorageError(f"Part checksum/byte count mismatch: {path}")
    return raw


def _manifest(root: Path):
    path = _safe_path(root, "manifest.json")
    if not path.exists():
        if any("/" in name for name in _owned_paths(root)):
            raise StorageError("Missing storage manifest for existing parts")
        return None
    try:
        value = json.loads(_file_bytes(root, "manifest.json"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"Invalid dataset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise StorageError("Dataset manifest must be an object")
    return value


def _storage(manifest):
    if manifest is None or "storage" not in manifest:
        return None
    storage = manifest["storage"]
    if not isinstance(storage, dict):
        raise StorageError("manifest.storage must be an object")
    if type(storage.get("version")) is not int or storage["version"] not in (1, STORAGE_VERSION):
        raise StorageError("Unsupported manifest.storage version")
    maximum = _integer(storage.get("max_file_bytes"), "storage.max_file_bytes", 1)
    assets = storage.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_FORMATS):
        raise StorageError("storage.assets must declare exactly the four logical assets")
    for logical, expected_format in ASSET_FORMATS.items():
        asset = assets[logical]
        if not isinstance(asset, dict) or asset.get("format") != expected_format:
            raise StorageError(f"Invalid storage format for {logical}")
        _integer(asset.get("rows"), f"{logical}.rows")
        _hash(asset.get("uncompressed_sha256"), f"{logical}.uncompressed_sha256")
        parts = asset.get("parts")
        if not isinstance(parts, list) or not parts:
            raise StorageError(f"{logical}.parts must be a nonempty list")
        for index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise StorageError(f"Invalid part descriptor for {logical}")
            digest = _hash(part.get("sha256"), f"{logical}.parts[{index}].sha256")
            if storage["version"] == 1:
                allowed = {_part_path(logical, index)}
                if len(parts) == 1:
                    allowed.add(logical)
            else:
                allowed = {_content_path(logical, digest)}
            path = part.get("path")
            if not isinstance(path, str) or path not in allowed:
                raise StorageError(f"Invalid or out-of-order part path for {logical}: {path!r}")
            _integer(part.get("rows"), f"{path}.rows")
            size = _integer(part.get("bytes"), f"{path}.bytes")
            if size > maximum:
                raise StorageError(f"Declared part exceeds max_file_bytes: {path}")
            if storage["version"] == STORAGE_VERSION:
                _integer(part.get("uncompressed_bytes"), f"{path}.uncompressed_bytes")
                if logical == "viewer_details.json.gz":
                    if part["uncompressed_bytes"] > maximum:
                        raise StorageError(f"Detail part exceeds uncompressed byte limit: {path}")
                    question_ids = part.get("question_ids")
                    if (not isinstance(question_ids, list)
                            or any(not isinstance(q, str) or not q.strip() or q != q.strip() for q in question_ids)
                            or question_ids != sorted(set(question_ids))
                            or (part["rows"] > 0 and len(question_ids) != 1)
                            or (part["rows"] == 0 and question_ids)):
                        raise StorageError(f"Invalid detail question_ids: {path}")
                elif "question_ids" in part:
                    raise StorageError(f"question_ids belong only on detail parts: {path}")
        paths = [part["path"] for part in parts]
        if len(set(paths)) != len(paths):
            raise StorageError(f"Duplicate part path for {logical}")
        if sum(part["rows"] for part in parts) != asset["rows"]:
            raise StorageError(f"Declared row count mismatch for {logical}")
    if len({asset["rows"] for asset in assets.values()}) != 1:
        raise StorageError("Logical asset row counts do not agree")
    return storage


def _reject_constant(value):
    raise StorageError(f"Non-JSON numeric constant: {value}")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _jsonl_items(raw: bytes, label: str) -> Iterator[tuple[dict | None, bytes]]:
    for line in io.BytesIO(raw):
        if not line.strip():
            yield None, line
            continue
        try:
            row = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StorageError(f"Invalid JSONL record in {label}") from exc
        if not isinstance(row, dict):
            raise StorageError(f"JSONL record is not an object in {label}")
        yield row, line


def _array_items(raw: bytes, label: str) -> Iterator[tuple[dict, bytes]]:
    """Parse objects while retaining their exact spelling and JSON whitespace."""
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise StorageError(f"Invalid UTF-8 array in {label}") from exc
    if not text.startswith("[") or not text.endswith("]"):
        raise StorageError(f"Expected JSON array without exterior whitespace in {label}")
    if text == "[]":
        return
    position = 1
    while position < len(text) - 1:
        start = position
        while position < len(text) - 1 and text[position] in " \t\r\n":
            position += 1
        try:
            row, position = _DECODER.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise StorageError(f"Invalid JSON array record in {label}") from exc
        if not isinstance(row, dict):
            raise StorageError(f"JSON array record is not an object in {label}")
        while position < len(text) - 1 and text[position] in " \t\r\n":
            position += 1
        row_bytes = text[start:position].encode("utf-8")
        if position == len(text) - 1:
            yield row, row_bytes
            return
        if text[position] != "," or position + 1 == len(text) - 1:
            raise StorageError(f"Invalid JSON array separator in {label}")
        yield row, row_bytes
        position += 1
    raise StorageError(f"Invalid empty/unterminated JSON array in {label}")


def _check_row(row: dict | None, seen: set[str], label: str) -> int:
    if row is None:
        return 0
    value = row.get("sample_id")
    if not isinstance(value, str) or not value.strip():
        raise StorageError(f"Missing/non-string/empty sample_id in {label}")
    if value != value.strip():
        raise StorageError(f"Surrounding whitespace in sample_id in {label}")
    sample_id = value.strip()
    if sample_id in seen:
        raise StorageError(f"Duplicate sample_id in {label}: {sample_id!r}")
    seen.add(sample_id)
    return 1


def _decompress(encoded: bytes, label: str) -> bytes:
    try:
        return gzip.decompress(encoded)
    except (OSError, EOFError, zlib.error) as exc:
        raise StorageError(f"Invalid gzip asset: {label}") from exc


def _question_id(row, label):
    value = row.get("question_id")
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StorageError(f"Missing/non-string/invalid question_id in {label}")
    return value


def _read_asset(root: Path, logical: str, asset, maximum: int, collect: bool,
                sample_ids=None, question_map=None, detail_questions=None):
    digest = hashlib.sha256()
    result = bytearray() if collect else None
    seen: set[str] = set()
    total_rows = 0

    def append(raw):
        digest.update(raw)
        if result is not None:
            result.extend(raw)

    is_array = asset["format"] == "json-gzip"
    if is_array:
        append(b"[")
    for index, part in enumerate(asset["parts"]):
        encoded = _file_bytes(root, part["path"], part, maximum)
        raw = _decompress(encoded, part["path"]) if is_array else encoded
        if "uncompressed_bytes" in part and len(raw) != part["uncompressed_bytes"]:
            raise StorageError(f"Part uncompressed byte count mismatch: {part['path']}")
        items = _array_items(raw, part["path"]) if is_array else _jsonl_items(raw, part["path"])
        rows = 0
        part_questions = set()
        for row, _ in items:
            rows += _check_row(row, seen, logical)
            if row is None:
                continue
            sample_id = row["sample_id"]
            if question_map is not None:
                question_map[sample_id] = _question_id(row, logical)
            if detail_questions is not None:
                if sample_id not in detail_questions:
                    raise StorageError(f"Detail sample_id missing from summary: {sample_id}")
                question_id = detail_questions[sample_id]
                if "question_id" in row and row["question_id"] != question_id:
                    raise StorageError(f"Detail question_id disagrees with summary: {sample_id}")
                part_questions.add(question_id)
        if detail_questions is not None and sorted(part_questions) != part.get("question_ids"):
            raise StorageError(f"Detail question_ids disagree with summary: {part['path']}")
        if rows != part["rows"]:
            raise StorageError(f"Part row count mismatch: {part['path']}")
        if is_array:
            if rows:
                if total_rows:
                    append(b",")
                append(raw[1:-1])
        else:
            if index < len(asset["parts"]) - 1 and raw and not raw.endswith(b"\n"):
                raise StorageError(f"JSONL part ends mid-line: {part['path']}")
            append(raw)
        total_rows += rows
    if is_array:
        append(b"]")
    if total_rows != asset["rows"]:
        raise StorageError(f"Logical row count mismatch: {logical}")
    if digest.hexdigest() != asset["uncompressed_sha256"]:
        raise StorageError(f"Logical checksum mismatch: {logical}")
    if sample_ids is not None:
        sample_ids.update(seen)
    return bytes(result) if result is not None else None


def _verify_storage(root: Path, storage) -> None:
    expected_ids = None
    question_map = {}
    for logical in ASSET_FORMATS:
        asset = storage["assets"][logical]
        sample_ids = set()
        v2 = storage["version"] == STORAGE_VERSION
        _read_asset(root, logical, asset, storage["max_file_bytes"], False, sample_ids,
                    question_map=question_map if v2 and logical == "viewer_rows.json.gz" else None,
                    detail_questions=question_map if v2 and logical == "viewer_details.json.gz" else None)
        if expected_ids is not None and sample_ids != expected_ids:
            raise StorageError(f"Logical asset sample_id sets do not agree: {logical}")
        expected_ids = sample_ids


def _logical_storage(path: Path):
    return _storage(_manifest(path.parent)) if path.name in ASSET_FORMATS else None


def read_bytes(logical_path) -> bytes:
    """Return exact *uncompressed* logical bytes, honoring declared storage.

    A monolithic shadow cannot bypass a storage declaration. Legacy JSONL files
    without storage are read directly, with no parsing or normalization.
    """
    path = Path(logical_path)
    storage = _logical_storage(path)
    if storage is not None:
        detail_questions = None
        if storage["version"] == STORAGE_VERSION and path.name == "viewer_details.json.gz":
            detail_questions = {}
            _read_asset(path.parent, "viewer_rows.json.gz", storage["assets"]["viewer_rows.json.gz"],
                        storage["max_file_bytes"], False, question_map=detail_questions)
        return _read_asset(path.parent, path.name, storage["assets"][path.name],
                            storage["max_file_bytes"], True, detail_questions=detail_questions)
    raw = _file_bytes(path.parent, path.name)
    return _decompress(raw, str(path)) if ASSET_FORMATS.get(path.name) == "json-gzip" else raw


def read_text(logical_path) -> str:
    """Read legacy or stored logical UTF-8 text without newline translation."""
    try:
        return read_bytes(logical_path).decode("utf-8")
    except UnicodeError as exc:
        raise StorageError(f"Invalid UTF-8 logical asset: {logical_path}") from exc


def asset_exists(logical_path) -> bool:
    """Check logical presence and file sizes without decoding the asset twice.

    Missing/invalid declarations fail closed. Content checksums and JSON are
    verified by read_bytes/read_text or dataset_exists, not by this stat check.
    """
    path = Path(logical_path)
    storage = _logical_storage(path)
    if storage is not None:
        for part in storage["assets"][path.name]["parts"]:
            target = _safe_path(path.parent, part["path"])
            try:
                info = target.stat()
            except FileNotFoundError as exc:
                raise StorageError(f"Missing dataset file: {target}") from exc
            if not stat.S_ISREG(info.st_mode) or info.st_size != part["bytes"]:
                raise StorageError(f"Part byte count/type mismatch: {target}")
        return True
    checked = _safe_path(path.parent, path.name)
    if not checked.exists():
        if path.name in ASSET_FORMATS:
            prefix = path.name.split(".", 1)[0] + "/"
            if any(name.startswith(prefix) for name in _owned_paths(path.parent)):
                raise StorageError(f"Missing storage manifest for existing parts: {path.name}")
        return False
    if not checked.is_file():
        raise StorageError(f"Dataset asset is not a regular file: {checked}")
    return True


def dataset_exists(output_dir) -> bool:
    """Validate an existing dataset; never mistake a broken one for a new one."""
    root = Path(output_dir)
    manifest = _manifest(root)
    storage = _storage(manifest)
    if storage is not None:
        _verify_storage(root, storage)
        return True
    present = {name for name in ASSET_FORMATS if _safe_path(root, name).exists()}
    orphaned = {
        name.split("/", 1)[0]
        for name in _owned_paths(root)
        if "/" in name
    } - {name.split(".", 1)[0] for name in present}
    if orphaned:
        raise StorageError("Missing storage manifest for existing parts")
    if not present and manifest is None:
        return False
    required = {"responses.jsonl", "aggregate.jsonl"}
    sources = (manifest or {}).get("sources", {})
    if not isinstance(sources, dict):
        raise StorageError("manifest.sources must be an object")
    for logical, key in SOURCE_KEYS.items():
        if key + "s" in sources:
            raise StorageError("Part-list sources require manifest.storage")
        if key in sources:
            required.add(logical)
    missing = required - present
    if missing:
        raise StorageError(f"Incomplete legacy dataset; missing: {', '.join(sorted(missing))}")
    expected_ids = None
    for logical in sorted(present):
        raw = read_bytes(root / logical)
        items = _array_items(raw, logical) if ASSET_FORMATS[logical] == "json-gzip" else _jsonl_items(raw, logical)
        seen: set[str] = set()
        for row, _ in items:
            _check_row(row, seen, logical)
        if expected_ids is not None and seen != expected_ids:
            raise StorageError(f"Logical asset sample_id sets do not agree: {logical}")
        expected_ids = seen
    return True


def _write_part(stage: Path, logical: str, encoded: bytes, rows: int, maximum: int,
                uncompressed_bytes: int, question_ids=None):
    digest = _sha256(encoded)
    relative = _content_path(logical, digest)
    if len(encoded) > maximum:
        raise StorageError(f"Individual record exceeds max_file_bytes: {relative}")
    path = _safe_path(stage, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    part = {"path": relative, "rows": rows, "bytes": len(encoded),
            "sha256": digest, "uncompressed_bytes": uncompressed_bytes}
    if question_ids is not None:
        part["question_ids"] = question_ids
    return part


def _gzip_batches(records: list[bytes], maximum: int, logical: str):
    raw = b"[" + b",".join(records) + b"]"
    encoded = _gzip_bytes(raw)
    if len(encoded) <= maximum:
        yield encoded, len(records), len(raw)
    elif len(records) <= 1:
        raise StorageError(f"Individual record exceeds max_file_bytes: {logical}")
    else:
        middle = len(records) // 2
        yield from _gzip_batches(records[:middle], maximum, logical)
        yield from _gzip_batches(records[middle:], maximum, logical)


def _pack_asset(stage: Path, logical: str, raw: bytes, maximum: int):
    is_array = ASSET_FORMATS[logical] == "json-gzip"
    seen: set[str] = set()
    rows = 0
    parts = []
    encoded = _gzip_bytes(raw) if is_array else raw
    items = _array_items(raw, logical) if is_array else _jsonl_items(raw, logical)
    if len(encoded) <= maximum:
        rows = sum(_check_row(row, seen, logical) for row, _ in items)
        parts.append(_write_part(stage, logical, encoded, rows, maximum, len(raw)))
    else:
        del encoded
        batch: list[bytes] = []
        batch_bytes = 2 if is_array else 0
        batch_rows = 0

        def flush():
            if is_array:
                chunks = _gzip_batches(batch, maximum, logical)
            else:
                chunks = [(b"".join(batch), batch_rows, sum(map(len, batch)))]
            for data, count, raw_bytes in chunks:
                parts.append(_write_part(stage, logical, data, count, maximum, raw_bytes))

        for row, record in items:
            count = _check_row(row, seen, logical)
            extra = len(record) + (1 if is_array and batch else 0)
            if batch and batch_bytes + extra > maximum:
                flush()
                batch = []
                batch_bytes = 2 if is_array else 0
                batch_rows = 0
            if not is_array and len(record) > maximum:
                raise StorageError(f"Individual record/line exceeds max_file_bytes: {logical}")
            batch_bytes += len(record) + (1 if is_array and batch else 0)
            batch.append(record)
            batch_rows += count
            rows += count
        if batch:
            flush()
        if not parts:
            raise StorageError(f"Empty asset cannot fit max_file_bytes: {logical}")
    return {
        "format": ASSET_FORMATS[logical],
        "rows": rows,
        "uncompressed_sha256": _sha256(raw),
        "parts": parts,
    }


def _pack_details(stage: Path, raw: bytes, question_map: dict[str, str], maximum: int):
    """Preserve detail records, partitioned by question for selective loading."""
    logical = "viewer_details.json.gz"
    groups: dict[str, list[tuple[str, bytes]]] = {}
    seen: set[str] = set()
    for row, record in _array_items(raw, logical):
        _check_row(row, seen, logical)
        sample_id = row["sample_id"]
        if sample_id not in question_map:
            raise StorageError(f"Detail sample_id missing from summary: {sample_id}")
        question_id = question_map[sample_id]
        if "question_id" in row and row["question_id"] != question_id:
            raise StorageError(f"Detail question_id disagrees with summary: {sample_id}")
        groups.setdefault(question_id, []).append((sample_id, record))
    if seen != set(question_map):
        raise StorageError("Detail sample_id sets do not agree with summary")

    parts = []
    digest = hashlib.sha256()
    digest.update(b"[")
    total = 0

    def flush(records, question_id):
        nonlocal total
        for encoded, count, raw_bytes in _gzip_batches(records, maximum, logical):
            # _gzip_batches may split incompressible framing overhead further.
            decoded = _decompress(encoded, logical)
            if count:
                if total:
                    digest.update(b",")
                digest.update(decoded[1:-1])
            total += count
            parts.append(_write_part(stage, logical, encoded, count, maximum, raw_bytes,
                                     [question_id] if count else []))

    for question_id in sorted(groups):
        batch = []
        size = 2
        for _, record in sorted(groups[question_id]):
            if len(record) + 2 > maximum:
                raise StorageError(f"Individual detail record exceeds max_file_bytes: {question_id}")
            extra = len(record) + bool(batch)
            if batch and size + extra > maximum:
                flush(batch, question_id)
                batch = []
                size = 2
            size += len(record) + bool(batch)
            batch.append(record)
        if batch:
            flush(batch, question_id)
    if not parts:
        flush([], "")
    digest.update(b"]")
    return {"format": "json-gzip", "rows": total,
            "uncompressed_sha256": digest.hexdigest(), "parts": parts}


def _source_prefix(manifest, logical: str) -> str:
    sources = manifest.get("sources", {})
    key = SOURCE_KEYS[logical]
    source = sources.get(key)
    if isinstance(source, str) and source.endswith(logical):
        return source[:-len(logical)]
    plural = sources.get(key + "s")
    if isinstance(plural, list) and plural and isinstance(plural[0], str):
        stem = logical.split(".", 1)[0] + "/"
        start = plural[0].rfind(stem)
        if start >= 0:
            return plural[0][:start]
    return ""


def _owned_paths(root: Path) -> set[str]:
    owned = set()
    for logical in ASSET_FORMATS:
        if _safe_path(root, logical).exists():
            owned.add(logical)
        stem, suffix = logical.split(".", 1)
        directory = _safe_path(root, stem)
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise StorageError(f"Expected generated part directory: {directory}")
        pattern = re.compile(r"(?:part-[0-9]{5,}|sha256-[0-9a-f]{64})\." + re.escape(suffix) + r"\Z")
        for path in directory.iterdir():
            if pattern.fullmatch(path.name):
                relative = f"{stem}/{path.name}"
                checked = _safe_path(root, relative)
                if not checked.is_file():
                    raise StorageError(f"Generated part is not a regular file: {checked}")
                owned.add(relative)
    return owned


def _install(root: Path, stage: Path, manifest, previous: set[str]) -> None:
    storage = _storage(manifest)
    _verify_storage(stage, storage)  # Full logical round trip before any replacement.
    targets = [part for asset in storage["assets"].values() for part in asset["parts"]]
    manifest_stage = stage / "manifest.json"
    with manifest_stage.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for part in targets:
        relative = part["path"]
        target = _safe_path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation never overwrites content an old manifest uses.
            os.link(_safe_path(stage, relative), target)
        except FileExistsError:
            _file_bytes(root, relative, part, storage["max_file_bytes"])
        _fsync_directory(target.parent)
    _verify_storage(root, storage)
    os.replace(manifest_stage, _safe_path(root, "manifest.json"))
    _fsync_directory(root)
    # Retain old hashes and legacy paths. A cached manifest can still read them,
    # and a failed activation only leaves harmless unreferenced content files.


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _dataset_lock(root: Path):
    """Serialize writers without creating a public lock-file artifact."""
    _safe_path(root, "manifest.json")
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def pack_dataset(dataset_dir, max_file_bytes: int = MAX_FILE_BYTES):
    """Pack fresh monoliths, or verify/repack an already complete stored dataset.

    Existing storage is authoritative over retained monoliths. A publisher can
    build fresh assets in an isolated directory with a manifest without storage.
    No old content file is removed. Version 1 migrates to version 2; an identical
    version 2 repeat verifies and returns without rewriting files.
    """
    maximum = _integer(max_file_bytes, "max_file_bytes", 1)
    root = Path(dataset_dir)
    with _dataset_lock(root):
        return _pack_dataset_locked(root, maximum)


def _pack_dataset_locked(root: Path, maximum: int):
    manifest = _manifest(root)
    if manifest is None:
        raise StorageError("pack requires the publisher's manifest.json")
    if not isinstance(manifest.get("sources", {}), dict):
        raise StorageError("manifest.sources must be an object")
    if not isinstance(manifest.get("counts", {}), dict):
        raise StorageError("manifest.counts must be an object")
    old_storage = _storage(manifest)
    fresh = old_storage is None
    if old_storage is not None:
        _verify_storage(root, old_storage)
        if old_storage["version"] == STORAGE_VERSION and maximum == old_storage["max_file_bytes"]:
            return manifest
    elif not all(_safe_path(root, name).is_file() for name in ASSET_FORMATS):
        raise StorageError("pack requires all four freshly generated monolithic assets")
    previous = _owned_paths(root)
    updated = copy.deepcopy(manifest)
    assets = {}
    question_map = {}
    with tempfile.TemporaryDirectory(prefix=".published-dataset-", dir=root) as temporary:
        stage = Path(temporary)
        for logical, fmt in ASSET_FORMATS.items():
            if fresh:
                # The publisher has written a fresh manifest without storage.
                # Read these monoliths explicitly, never old shard contents.
                raw = _file_bytes(root, logical)
                if fmt == "json-gzip":
                    raw = _decompress(raw, logical)
            else:
                raw = _read_asset(root, logical, old_storage["assets"][logical], old_storage["max_file_bytes"], True)
            if logical == "viewer_rows.json.gz":
                summary_ids = set()
                for row, _ in _array_items(raw, logical):
                    _check_row(row, summary_ids, logical)
                    question_map[row["sample_id"]] = _question_id(row, logical)
            if logical == "viewer_details.json.gz":
                assets[logical] = _pack_details(stage, raw, question_map, maximum)
            else:
                assets[logical] = _pack_asset(stage, logical, raw, maximum)
            del raw
        updated["storage"] = {"version": STORAGE_VERSION, "max_file_bytes": maximum, "assets": assets}
        sources = updated.setdefault("sources", {})
        for logical, key in SOURCE_KEYS.items():
            paths = [_source_prefix(manifest, logical) + part["path"] for part in assets[logical]["parts"]]
            sources.pop(key, None)
            sources.pop(key + "s", None)
            if len(paths) == 1 and assets[logical]["parts"][0]["path"] == logical:
                sources[key] = paths[0]
            else:
                sources[key + "s"] = paths
        counts = updated.setdefault("counts", {})
        counts["responses_rows"] = assets["responses.jsonl"]["rows"]
        counts["aggregate_rows"] = assets["aggregate.jsonl"]["rows"]
        for stem in ("viewer_rows", "viewer_details"):
            counts[stem + "_bytes"] = sum(part["bytes"] for part in assets[stem + ".json.gz"]["parts"])
        _install(root, stage, updated, previous)
    return updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="pack and verify freshly published assets")
    pack.add_argument("dataset_dir", type=Path)
    pack.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    cat = commands.add_parser("cat", help="emit exact uncompressed logical bytes")
    cat.add_argument("logical_path", type=Path)
    verify = commands.add_parser("verify", help="verify every declared asset")
    verify.add_argument("dataset_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "cat":
            sys.stdout.buffer.write(read_bytes(args.logical_path))
        elif args.command == "verify":
            if not dataset_exists(args.dataset_dir):
                raise StorageError(f"No dataset exists: {args.dataset_dir}")
            print(f"Verified dataset: {args.dataset_dir}")
        else:
            manifest = pack_dataset(args.dataset_dir, args.max_file_bytes)
            parts = sum(len(asset["parts"]) for asset in manifest["storage"]["assets"].values())
            print(f"Packed {len(ASSET_FORMATS)} assets into {parts} parts (limit {args.max_file_bytes} bytes)")
    except (StorageError, OSError) as exc:
        print(f"published_dataset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
