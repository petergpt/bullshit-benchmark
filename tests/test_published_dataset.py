import copy
import gzip
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "published_dataset.py"
SPEC = importlib.util.spec_from_file_location("published_dataset", SCRIPT)
STORAGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STORAGE)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def load_consumer(stem):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "scripts" / (stem + ".py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, {"published_dataset": STORAGE, stem: module}):
        spec.loader.exec_module(module)
    return module


class PublishedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="published-dataset-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)

    def rows(self, count=24):
        rows = []
        for index in range(count):
            varied = "".join(sha256(f"{index}-{part}".encode()) for part in range(4))
            rows.append({
                "sample_id": f"sample-{index:03}",
                "question_id": f"question-{index // 4:03}",
                "response_text": None if index == 1 else f"café 雪 🙂\u2028\n{varied}",
                "response_refusal": index == 1,
                "response_usage": {"completion_tokens": index, "reasoning_tokens": None},
                "untouched": {"arbitrary_key": varied, "nullable": None},
            })
        return rows

    def write_manifest(self, manifest):
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def load_manifest(self):
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def write_fresh(self, raws, preserve_manifest=False):
        for name, raw in raws.items():
            encoded = gzip.compress(raw, mtime=123) if name.endswith(".gz") else raw
            (self.root / name).write_bytes(encoded)
        if not preserve_manifest:
            self.write_manifest({
                "sources": {key: "data/latest/" + name for name, key in STORAGE.SOURCE_KEYS.items()},
                "counts": {},
                "keep_metadata": {"nested": [1, None]},
            })

    def fixture(self, count=24):
        rows = self.rows(count)
        jsonl = b"".join(compact(row) + (b"\r\n" if index % 2 else b"\n") for index, row in enumerate(rows))
        # Blank lines, CRLF, Unicode, nulls, and unknown fields are all preserved.
        jsonl = b"\n" + jsonl + b"\r\n"
        raws = {
            "responses.jsonl": jsonl,
            "aggregate.jsonl": jsonl,
            "viewer_rows.json.gz": compact(rows),
            "viewer_details.json.gz": compact(rows),
        }
        self.write_fresh(raws)
        return raws

    def pack(self, maximum=1024):
        return STORAGE.pack_dataset(self.root, maximum)

    def consumer_fixture(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            row.update({
                "model_id": f"example/model-{index % 2}",
                "model": f"example/model-{index % 2}@reasoning=low",
                "model_reasoning_level": "low",
                "question_id": f"question-{index // 2:03}",
                "question": "A full-schema question with literal café 雪\u2028separator?",
                "domain": "fixture",
                "technique": "fixture",
                "consensus_score": None if index == 1 else index % 3,
                "row_errors": [],
                "row_identity_mismatch": False,
            })
        jsonl = b"".join(compact(row) + b"\n" for row in rows)
        self.write_fresh({
            "responses.jsonl": jsonl,
            "aggregate.jsonl": jsonl,
            "viewer_rows.json.gz": compact(rows),
            "viewer_details.json.gz": compact(rows),
        })
        return rows

    def asset(self, name="responses.jsonl"):
        return self.load_manifest()["storage"]["assets"][name]

    def first_part(self, name="responses.jsonl"):
        return self.root / self.asset(name)["parts"][0]["path"]

    def snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def assert_roundtrip(self, raws):
        self.assertTrue(STORAGE.dataset_exists(self.root))
        for name, raw in raws.items():
            with self.subTest(name=name):
                self.assertTrue(STORAGE.asset_exists(self.root / name))
                self.assertEqual(STORAGE.read_bytes(self.root / name), raw)
                self.assertEqual(STORAGE.read_text(self.root / name).encode("utf-8"), raw)

    def test_legacy_read_does_not_parse_or_translate_text(self):
        raw = "legacy text, not JSON\r\n雪\n".encode("utf-8")
        path = self.root / "responses.jsonl"
        path.write_bytes(raw)
        self.assertEqual(STORAGE.read_text(path).encode("utf-8"), raw)
        self.assertTrue(STORAGE.asset_exists(path))
        self.assertFalse(STORAGE.asset_exists(self.root / "aggregate.jsonl"))

    def test_dataset_absence_and_incomplete_legacy_are_distinct(self):
        self.assertFalse(STORAGE.dataset_exists(self.root))
        (self.root / "responses.jsonl").write_bytes(b'{"sample_id":"sample"}\n')
        with self.assertRaisesRegex(STORAGE.StorageError, "Incomplete legacy"):
            STORAGE.dataset_exists(self.root)
        (self.root / "aggregate.jsonl").write_bytes(b'{"sample_id":"sample"}\n')
        self.assertTrue(STORAGE.dataset_exists(self.root))
        self.write_manifest({"sources": {"viewer_rows_file": "data/latest/viewer_rows.json.gz"}})
        with self.assertRaisesRegex(STORAGE.StorageError, "Incomplete legacy"):
            STORAGE.dataset_exists(self.root)

    def test_jsonl_roundtrip_preserves_every_byte_and_key(self):
        raws = self.fixture()
        manifest = self.pack()
        self.assert_roundtrip(raws)
        for logical in ("responses.jsonl", "aggregate.jsonl"):
            asset = manifest["storage"]["assets"][logical]
            self.assertGreater(len(asset["parts"]), 1)
            combined = b"".join((self.root / part["path"]).read_bytes() for part in asset["parts"])
            self.assertEqual(combined, raws[logical])
            self.assertTrue((self.root / logical).exists())
        row = [json.loads(line) for line in STORAGE.read_bytes(self.root / "responses.jsonl").splitlines() if line.strip()][1]
        self.assertIsNone(row["response_text"])
        self.assertTrue(row["response_refusal"])
        self.assertIsNone(row["response_usage"]["reasoning_tokens"])
        self.assertIn("arbitrary_key", row["untouched"])

    def test_gzip_shards_preserve_numeric_spelling_and_array_interiors(self):
        raws = self.fixture()
        pieces = []
        for row in self.rows():
            body = compact(row)
            pieces.append(body[:-1] + b',"float":1e+00,"negative_zero":-0.0,"large":9007199254740993}')
        raw = b"[" + b", ".join(pieces) + b"]"
        raws["viewer_rows.json.gz"] = raw
        self.write_fresh(raws)
        manifest = self.pack()
        asset = manifest["storage"]["assets"]["viewer_rows.json.gz"]
        self.assertGreater(len(asset["parts"]), 1)
        interiors = [gzip.decompress((self.root / part["path"]).read_bytes())[1:-1] for part in asset["parts"]]
        self.assertEqual(b"[" + b",".join(interiors) + b"]", raw)
        self.assertEqual(asset["uncompressed_sha256"], sha256(raw))
        self.assert_roundtrip(raws)

    def test_small_assets_use_immutable_paths_with_deterministic_gzip(self):
        raws = self.fixture(2)
        manifest = self.pack(100_000)
        for logical, asset in manifest["storage"]["assets"].items():
            self.assertEqual(len(asset["parts"]), 1)
            self.assertEqual(asset["parts"][0]["path"], STORAGE._content_path(logical, asset["parts"][0]["sha256"]))
            key = STORAGE.SOURCE_KEYS[logical]
            self.assertEqual(manifest["sources"][key + "s"], ["data/latest/" + asset["parts"][0]["path"]])
            self.assertNotIn(key, manifest["sources"])
            if logical.endswith(".gz"):
                encoded = (self.root / asset["parts"][0]["path"]).read_bytes()
                self.assertEqual(encoded[4:8], b"\0" * 4)
                self.assertEqual(encoded[9], 255)
        self.assertEqual(manifest["keep_metadata"], {"nested": [1, None]})
        self.assert_roundtrip(raws)

    def test_gzip_is_deterministic_across_fresh_publications(self):
        raws = self.fixture()
        first_manifest = self.pack()
        first = self.snapshot()
        self.write_fresh(raws)  # Publisher resets the manifest before pack.
        second_manifest = self.pack()
        self.assertEqual(second_manifest, first_manifest)
        self.assertEqual(self.snapshot(), first)

    def test_every_part_obeys_limit_counts_hashes_and_sources(self):
        self.fixture()
        manifest = self.pack()
        for logical, asset in manifest["storage"]["assets"].items():
            self.assertEqual(asset["rows"], 24)
            key = STORAGE.SOURCE_KEYS[logical]
            self.assertNotIn(key, manifest["sources"])
            self.assertEqual(manifest["sources"][key + "s"], ["data/latest/" + p["path"] for p in asset["parts"]])
            for index, part in enumerate(asset["parts"]):
                raw = (self.root / part["path"]).read_bytes()
                self.assertLessEqual(len(raw), 1024)
                self.assertEqual(part["bytes"], len(raw))
                self.assertEqual(part["sha256"], sha256(raw))
                self.assertIn("sha256-" + part["sha256"], part["path"])
            if logical.endswith(".gz"):
                key = logical.split(".")[0] + "_bytes"
                self.assertEqual(manifest["counts"][key], sum(p["bytes"] for p in asset["parts"]))

    def test_oversize_jsonl_record_leaves_originals_untouched(self):
        self.fixture()
        (self.root / "responses.jsonl").write_bytes(compact({"sample_id": "large", "text": "x" * 5000}) + b"\n")
        before = self.snapshot()
        with self.assertRaisesRegex(STORAGE.StorageError, "Individual record"):
            self.pack()
        self.assertEqual(self.snapshot(), before)

    def test_oversize_gzip_record_leaves_originals_untouched(self):
        self.fixture(1)
        varied = "".join(sha256(str(index).encode()) for index in range(100))
        raw = compact([{"sample_id": "sample-000", "text": varied}])
        (self.root / "viewer_details.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        before = self.snapshot()
        with self.assertRaisesRegex(STORAGE.StorageError, "Individual detail record"):
            self.pack()
        self.assertEqual(self.snapshot(), before)

    def test_compact_summary_can_exceed_uncompressed_limit_when_encoded_small(self):
        raws = self.fixture(1)
        large = compact([{"sample_id": "sample-000", "question_id": "question-000", "text": "x" * 30_000}])
        raws["viewer_rows.json.gz"] = large
        self.write_fresh(raws)
        self.pack()
        parts = self.asset("viewer_rows.json.gz")["parts"]
        self.assertEqual(len(parts), 1)
        self.assertLessEqual(parts[0]["bytes"], 1024)
        self.assertGreater(parts[0]["uncompressed_bytes"], 1024)
        self.assert_roundtrip(raws)

    def test_empty_assets_roundtrip_and_tiny_cap_fails_safely(self):
        raws = self.fixture(0)
        self.pack(100)
        self.assert_roundtrip(raws)
        before = self.snapshot()
        with self.assertRaises(STORAGE.StorageError):
            self.pack(10)
        self.assertEqual(self.snapshot(), before)

    def test_missing_part_fails_all_public_read_gates(self):
        self.fixture()
        self.pack()
        self.first_part().unlink()
        for operation in (
            lambda: STORAGE.dataset_exists(self.root),
            lambda: STORAGE.asset_exists(self.root / "responses.jsonl"),
            lambda: STORAGE.read_text(self.root / "responses.jsonl"),
            self.pack,
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(STORAGE.StorageError, "Missing dataset file"):
                operation()

    def test_corrupt_part_same_size_fails_checksum(self):
        self.fixture()
        self.pack()
        path = self.first_part()
        raw = bytearray(path.read_bytes())
        raw[-3] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(STORAGE.StorageError, "checksum"):
            STORAGE.dataset_exists(self.root)

    def test_corrupt_manifest_counts_hashes_and_formats_fail(self):
        self.fixture()
        original = self.pack()
        mutations = (
            lambda m: m["storage"]["assets"]["responses.jsonl"].update(uncompressed_sha256="0" * 64),
            lambda m: m["storage"]["assets"]["responses.jsonl"]["parts"][0].update(sha256="0" * 64),
            lambda m: m["storage"]["assets"]["responses.jsonl"]["parts"][0].update(bytes=1),
            lambda m: m["storage"]["assets"]["responses.jsonl"]["parts"][0].update(rows=999),
            lambda m: m["storage"]["assets"]["responses.jsonl"].update(rows=999),
            lambda m: m["storage"]["assets"]["responses.jsonl"].update(format="json-gzip"),
            lambda m: m["storage"]["assets"]["viewer_rows.json.gz"].update(format="jsonl"),
            lambda m: m["storage"]["assets"].pop("viewer_details.json.gz"),
            lambda m: m["storage"]["assets"].update(unexpected={}),
            lambda m: m["storage"].update(max_file_bytes=True),
        )
        for mutate in mutations:
            manifest = copy.deepcopy(original)
            mutate(manifest)
            self.write_manifest(manifest)
            with self.subTest(mutation=mutate), self.assertRaises(STORAGE.StorageError):
                STORAGE.dataset_exists(self.root)

    def test_actual_row_count_is_checked_not_only_manifest_totals(self):
        self.fixture()
        manifest = self.pack()
        asset = manifest["storage"]["assets"]["responses.jsonl"]
        asset["parts"][0]["rows"] += 1
        asset["parts"][1]["rows"] -= 1
        self.write_manifest(manifest)
        with self.assertRaisesRegex(STORAGE.StorageError, "Part row count"):
            STORAGE.read_bytes(self.root / "responses.jsonl")

    def test_unknown_or_malformed_storage_version_fails_closed(self):
        self.fixture()
        original = self.pack()
        for version in (None, 0, 3, "1", True):
            manifest = copy.deepcopy(original)
            manifest["storage"]["version"] = version
            self.write_manifest(manifest)
            with self.subTest(version=version), self.assertRaisesRegex(STORAGE.StorageError, "version"):
                STORAGE.dataset_exists(self.root)
        for storage in (None, [], {}):
            manifest = copy.deepcopy(original)
            manifest["storage"] = storage
            self.write_manifest(manifest)
            with self.subTest(storage=storage), self.assertRaises(STORAGE.StorageError):
                STORAGE.dataset_exists(self.root)

    def test_traversal_absolute_url_encoded_and_reordered_paths_fail(self):
        self.fixture()
        original = self.pack()
        for path in (
            "../outside.jsonl", "/tmp/outside.jsonl", "https://example.invalid/data.jsonl",
            "responses/../outside.jsonl", "responses//part-00000.jsonl",
            "responses/%2e%2e/outside.jsonl", "responses\\part-00000.jsonl",
            "responses/part-00000.jsonl?download=1", "responses/./part-00000.jsonl",
            "aggregate/part-00000.jsonl", "responses/part-00001.jsonl",
        ):
            manifest = copy.deepcopy(original)
            manifest["storage"]["assets"]["responses.jsonl"]["parts"][0]["path"] = path
            self.write_manifest(manifest)
            with self.subTest(path=path), self.assertRaisesRegex(STORAGE.StorageError, "part path"):
                STORAGE.dataset_exists(self.root)

    def test_part_file_and_directory_symlinks_are_rejected(self):
        self.fixture()
        self.pack()
        part = self.first_part()
        backup = self.root / "unrelated-copy.jsonl"
        part.rename(backup)
        part.symlink_to(backup)
        with self.assertRaisesRegex(STORAGE.StorageError, "Symlink"):
            STORAGE.dataset_exists(self.root)
        part.unlink()
        backup.rename(part)
        directory = part.parent
        moved = self.root / "unrelated-directory"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(STORAGE.StorageError, "Symlink"):
            STORAGE.read_bytes(self.root / "responses.jsonl")

    def test_invalid_gzip_rejected_even_with_updated_encoded_checksum(self):
        self.fixture()
        manifest = self.pack()
        asset = manifest["storage"]["assets"]["viewer_details.json.gz"]
        part = asset["parts"][0]
        raw = b"not a gzip stream"
        part.update(bytes=len(raw), sha256=sha256(raw), path=STORAGE._content_path("viewer_details.json.gz", sha256(raw)))
        (self.root / part["path"]).write_bytes(raw)
        self.write_manifest(manifest)
        with self.assertRaisesRegex(STORAGE.StorageError, "Invalid gzip"):
            STORAGE.dataset_exists(self.root)

    def test_repeat_and_limit_migrations_do_not_lose_data(self):
        raws = self.fixture()
        original = self.pack()
        before = self.snapshot()
        self.assertEqual(self.pack(), original)
        self.assertEqual(self.snapshot(), before)
        note = self.root / "responses" / "notes.txt"
        note.write_text("not a generated part", encoding="utf-8")
        stale = self.root / "responses" / "part-99999.jsonl"
        stale.write_text("old generated part", encoding="utf-8")
        self.pack(100_000)
        self.assert_roundtrip(raws)
        self.assertTrue(note.exists())
        self.assertTrue(stale.exists())
        self.assertEqual(set(self.load_manifest()["sources"]), {key + "s" for key in STORAGE.SOURCE_KEYS.values()})
        self.pack(1024)
        self.assert_roundtrip(raws)

    def test_new_records_only_change_final_existing_jsonl_shard(self):
        raws = self.fixture()
        manifest = self.pack()
        old_parts = manifest["storage"]["assets"]["responses.jsonl"]["parts"]
        old_bytes = [(self.root / part["path"]).read_bytes() for part in old_parts]
        extra = self.rows(25)[-1]
        for logical in raws:
            if logical.endswith(".gz"):
                raws[logical] = raws[logical][:-1] + b"," + compact(extra) + b"]"
            else:
                raws[logical] += compact(extra) + b"\n"
        self.write_fresh(raws)
        newer = self.pack()
        new_parts = newer["storage"]["assets"]["responses.jsonl"]["parts"]
        for index, old in enumerate(old_bytes[:-1]):
            self.assertEqual((self.root / new_parts[index]["path"]).read_bytes(), old)
        self.assert_roundtrip(raws)

    def test_duplicate_sample_ids_fail_before_replacement(self):
        self.fixture()
        path = self.root / "responses.jsonl"
        path.write_bytes(path.read_bytes() + compact(self.rows(1)[0]) + b"\n")
        before = self.snapshot()
        with self.assertRaisesRegex(STORAGE.StorageError, "Duplicate sample_id"):
            self.pack()
        self.assertEqual(self.snapshot(), before)

    def test_missing_empty_and_nonstring_sample_ids_are_rejected(self):
        raws = self.fixture(1)
        for value in (None, "", "  ", " sample-000", "sample-000 ", 123, False, [], {}):
            changed = dict(raws)
            changed["viewer_details.json.gz"] = compact([{"sample_id": value}])
            self.write_fresh(changed)
            before = self.snapshot()
            with self.subTest(value=value), self.assertRaisesRegex(STORAGE.StorageError, "sample_id"):
                self.pack()
            self.assertEqual(self.snapshot(), before)

    def test_duplicate_ids_across_stored_parts_are_rejected(self):
        self.fixture()
        manifest = self.pack()
        asset = manifest["storage"]["assets"]["responses.jsonl"]
        second = asset["parts"][1]
        path = self.root / second["path"]
        raw = path.read_bytes().replace(b"sample-001", b"sample-000")
        self.assertNotEqual(raw, path.read_bytes())
        second.update(bytes=len(raw), sha256=sha256(raw), path=STORAGE._content_path("responses.jsonl", sha256(raw)))
        (self.root / second["path"]).write_bytes(raw)
        all_raw = b"".join((self.root / part["path"]).read_bytes() for part in asset["parts"])
        asset["uncompressed_sha256"] = sha256(all_raw)
        self.write_manifest(manifest)
        with self.assertRaisesRegex(STORAGE.StorageError, "Duplicate sample_id"):
            STORAGE.dataset_exists(self.root)

    def test_cross_asset_id_mismatch_rejected_before_install(self):
        raws = self.fixture()
        raws["viewer_details.json.gz"] = raws["viewer_details.json.gz"].replace(b"sample-000", b"sample-999")
        self.write_fresh(raws)
        before = self.snapshot()
        with self.assertRaisesRegex(STORAGE.StorageError, "sample_id"):
            self.pack()
        self.assertEqual(self.snapshot(), before)

    def test_valid_json_truncation_of_small_packed_asset_cannot_be_blessed(self):
        self.fixture(2)
        self.pack(100_000)
        path = self.first_part("responses.jsonl")
        path.write_bytes(compact(self.rows(1)[0]) + b"\n")
        before = self.snapshot()
        with self.assertRaises(STORAGE.StorageError):
            self.pack(100_000)
        self.assertEqual(self.snapshot(), before)

    def test_stale_monolithic_shadows_never_override_declared_shards(self):
        raws = self.fixture()
        self.pack()
        self.write_fresh(raws, preserve_manifest=True)
        (self.root / "responses.jsonl").write_bytes(b"\n")
        self.assertEqual(STORAGE.read_bytes(self.root / "responses.jsonl"), raws["responses.jsonl"])
        before = self.snapshot()
        self.pack()
        self.assertEqual(self.snapshot(), before)

    def test_orphaned_parts_without_manifest_are_not_a_new_dataset(self):
        self.fixture()
        self.pack()
        for name in STORAGE.ASSET_FORMATS:
            (self.root / name).unlink(missing_ok=True)
        (self.root / "manifest.json").unlink()
        with self.assertRaisesRegex(STORAGE.StorageError, "Missing storage manifest"):
            STORAGE.dataset_exists(self.root)
        for logical in STORAGE.ASSET_FORMATS:
            with self.subTest(logical=logical), self.assertRaisesRegex(STORAGE.StorageError, "Missing storage manifest"):
                STORAGE.asset_exists(self.root / logical)

    def test_orphaned_parts_with_all_monolithic_shadows_still_require_manifest(self):
        raws = self.fixture()
        self.pack()
        self.write_fresh(raws, preserve_manifest=True)
        (self.root / "responses.jsonl").write_bytes(b"\n")
        (self.root / "manifest.json").unlink()
        before = self.snapshot()
        for operation in (
            lambda: STORAGE.dataset_exists(self.root),
            lambda: STORAGE.asset_exists(self.root / "responses.jsonl"),
            lambda: STORAGE.read_text(self.root / "responses.jsonl"),
            self.pack,
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(STORAGE.StorageError, "Missing storage manifest"):
                operation()
        self.assertEqual(self.snapshot(), before)

    def test_failure_before_manifest_activation_keeps_old_dataset_readable(self):
        raws = self.fixture()
        self.pack(1500)
        before = self.snapshot()
        actual_replace = STORAGE.os.replace

        def fail_manifest(source, target):
            self.assert_roundtrip(raws)
            if pathlib.Path(target).name == "manifest.json":
                raise OSError("simulated activation failure")
            return actual_replace(source, target)

        with mock.patch.object(STORAGE.os, "replace", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "simulated activation failure"):
                self.pack(2200)
        self.assert_roundtrip(raws)
        after = self.snapshot()
        for path, raw in before.items():
            self.assertEqual(after[path], raw, path)

    def test_reader_holding_previous_manifest_can_read_during_and_after_install(self):
        raws = self.fixture()
        previous = self.pack(1500)
        actual_link = STORAGE.os.link
        observed = []

        def observe_install(source, target):
            result = actual_link(source, target)
            if not observed:
                STORAGE._verify_storage(self.root, previous["storage"])
                self.assertEqual(self.load_manifest(), previous)
                observed.append(True)
            return result

        with mock.patch.object(STORAGE.os, "link", side_effect=observe_install):
            self.pack(2200)
        self.assertTrue(observed)
        STORAGE._verify_storage(self.root, previous["storage"])
        self.assert_roundtrip(raws)

    def test_legacy_cross_asset_id_mismatch_fails_verification(self):
        raws = self.fixture(2)
        raws["aggregate.jsonl"] = raws["aggregate.jsonl"].replace(b"sample-000", b"sample-999")
        self.write_fresh(raws)
        with self.assertRaisesRegex(STORAGE.StorageError, "sample_id sets"):
            STORAGE.dataset_exists(self.root)

    def test_v1_manifest_reads_and_migrates_without_changing_canonical_bytes(self):
        raws = self.fixture(4)
        manifest = self.load_manifest()
        assets = {}
        for name, raw in raws.items():
            encoded = (self.root / name).read_bytes()
            assets[name] = {
                "format": STORAGE.ASSET_FORMATS[name], "rows": 4,
                "uncompressed_sha256": sha256(raw),
                "parts": [{"path": name, "rows": 4, "bytes": len(encoded), "sha256": sha256(encoded)}],
            }
        manifest["storage"] = {"version": 1, "max_file_bytes": 100_000, "assets": assets}
        self.write_manifest(manifest)
        self.assert_roundtrip(raws)
        migrated = self.pack(1500)
        self.assertEqual(migrated["storage"]["version"], 2)
        self.assert_roundtrip(raws)
        # The v1 manifest still names files that remain readable after activation.
        STORAGE._verify_storage(self.root, manifest["storage"])

    def test_details_group_by_question_preserving_every_record_and_canonical_byte(self):
        rows = self.rows(12)
        for index, row in enumerate(rows):
            row["question_id"] = f"question-{index % 3}"
        rows.reverse()
        jsonl = b"".join(compact(row) + b"\n" for row in rows)
        raws = {"responses.jsonl": jsonl, "aggregate.jsonl": jsonl,
                "viewer_rows.json.gz": compact(rows), "viewer_details.json.gz": compact(rows)}
        self.write_fresh(raws)
        manifest = self.pack(1500)
        for name in ("responses.jsonl", "aggregate.jsonl", "viewer_rows.json.gz"):
            self.assertEqual(STORAGE.read_bytes(self.root / name), raws[name])
        details = json.loads(STORAGE.read_bytes(self.root / "viewer_details.json.gz"))
        self.assertEqual(details, sorted(rows, key=lambda row: (row["question_id"], row["sample_id"])))
        by_id = {row["sample_id"]: row for row in rows}
        for part in manifest["storage"]["assets"]["viewer_details.json.gz"]["parts"]:
            parsed = json.loads(gzip.decompress((self.root / part["path"]).read_bytes()))
            self.assertEqual(len(part["question_ids"]), 1)
            self.assertLessEqual(part["uncompressed_bytes"], 1500)
            self.assertLessEqual(part["bytes"], 1500)
            for row in parsed:
                self.assertEqual(row, by_id[row["sample_id"]])
                self.assertEqual([row["question_id"]], part["question_ids"])
        self.assertTrue(STORAGE.dataset_exists(self.root))

    def test_question_partition_metadata_must_match_summary(self):
        self.fixture(4)
        manifest = self.pack()
        part = manifest["storage"]["assets"]["viewer_details.json.gz"]["parts"][0]
        part["question_ids"] = ["wrong-question"]
        self.write_manifest(manifest)
        with self.assertRaisesRegex(STORAGE.StorageError, "question_ids disagree"):
            STORAGE.dataset_exists(self.root)
        with self.assertRaisesRegex(STORAGE.StorageError, "question_ids disagree"):
            STORAGE.read_bytes(self.root / "viewer_details.json.gz")

    def test_missing_summary_question_id_and_disagreeing_detail_question_fail(self):
        raws = self.fixture(1)
        for name, row in (
            ("viewer_rows.json.gz", {"sample_id": "sample-000"}),
            ("viewer_details.json.gz", {"sample_id": "sample-000", "question_id": "wrong"}),
        ):
            changed = dict(raws, **{name: compact([row])})
            self.write_fresh(changed)
            with self.subTest(name=name), self.assertRaisesRegex(STORAGE.StorageError, "question_id"):
                self.pack()

    def test_unchanged_question_groups_reuse_existing_content_files(self):
        raws = self.fixture(12)
        previous = self.pack(5000)
        old_parts = previous["storage"]["assets"]["viewer_details.json.gz"]["parts"]
        extra = dict(self.rows(13)[-1], question_id="question-000")
        for logical in raws:
            if logical.endswith(".gz"):
                raws[logical] = raws[logical][:-1] + b"," + compact(extra) + b"]"
            else:
                raws[logical] += compact(extra) + b"\n"
        self.write_fresh(raws)
        newer = self.pack(5000)
        new_parts = newer["storage"]["assets"]["viewer_details.json.gz"]["parts"]
        for part in old_parts:
            if part["question_ids"] != ["question-000"]:
                self.assertIn(part, new_parts)
                self.assertTrue((self.root / part["path"]).exists())
        before = self.snapshot()
        self.assertEqual(self.pack(5000), newer)
        self.assertEqual(self.snapshot(), before)

    def test_existence_check_does_not_decode_or_hash_entire_asset_again(self):
        self.fixture(4)
        self.pack()
        with mock.patch.object(STORAGE, "_read_asset", side_effect=AssertionError("unexpected full read")):
            self.assertTrue(STORAGE.asset_exists(self.root / "responses.jsonl"))
        self.first_part().unlink()
        with self.assertRaisesRegex(STORAGE.StorageError, "Missing dataset file"):
            STORAGE.asset_exists(self.root / "responses.jsonl")

    def test_v2_uncompressed_byte_counts_are_verified(self):
        self.fixture(4)
        manifest = self.pack()
        part = manifest["storage"]["assets"]["viewer_rows.json.gz"]["parts"][0]
        part["uncompressed_bytes"] += 1
        self.write_manifest(manifest)
        with self.assertRaisesRegex(STORAGE.StorageError, "uncompressed byte count"):
            STORAGE.dataset_exists(self.root)

    def test_cli_cat_exact_bytes_and_corruption_goes_to_stderr(self):
        raws = self.fixture()
        completed = subprocess.run([sys.executable, str(SCRIPT), "pack", str(self.root), "--max-file-bytes", "1024"], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for logical in ("responses.jsonl", "viewer_details.json.gz"):
            completed = subprocess.run([sys.executable, str(SCRIPT), "cat", str(self.root / logical)], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, raws[logical])
        self.first_part().unlink()
        completed = subprocess.run([sys.executable, str(SCRIPT), "cat", str(self.root / "responses.jsonl")], capture_output=True)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"Missing dataset file", completed.stderr)

    def test_model_launch_jsonl_reader_matches_legacy_with_unicode_separator(self):
        expected = self.consumer_fixture()
        consumer = load_consumer("model_launch_pipeline")
        path = self.root / "responses.jsonl"
        self.assertEqual(consumer.read_jsonl(path), expected)
        self.pack(2048)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        self.assertEqual(consumer.read_jsonl(path), expected)

    def test_model_launch_inventory_matches_legacy_and_rejects_missing_shards(self):
        self.consumer_fixture()
        consumer = load_consumer("model_launch_pipeline")
        config = self.root / "fixture-config.json"
        config.write_text(json.dumps({"collect": {
            "models": ["example/model-0"],
            "model_reasoning_efforts": {"example/model-0": ["low"]},
        }}), encoding="utf-8")
        args = (config, self.root / "aggregate.jsonl", self.root / "responses.jsonl", self.root / "absent-run-history")
        expected = consumer.scan_inventory(*args)
        self.assertEqual([row["model_id"] for row in expected[0]], ["example/model-0", "example/model-1"])
        self.pack(2048)
        self.assertEqual(consumer.scan_inventory(*args), expected)
        self.first_part().unlink()
        with self.assertRaisesRegex(STORAGE.StorageError, "Missing dataset file"):
            consumer.scan_inventory(*args)

    def test_forge_aggregate_reader_matches_legacy_full_schema(self):
        self.consumer_fixture()
        consumer = load_consumer("push_bullshitbench_to_forge")
        path = self.root / "aggregate.jsonl"
        expected = consumer.load_aggregate_rows(path)
        self.assertEqual(sum(len(rows) for rows in expected.values()), 23)
        self.assertIn("\u2028", expected["question-000"][0].question)
        self.pack(2048)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        self.assertEqual(consumer.load_aggregate_rows(path), expected)
        self.first_part("aggregate.jsonl").unlink()
        with self.assertRaisesRegex(consumer.ForgeError, "Missing dataset file"):
            consumer.load_aggregate_rows(path)


if __name__ == "__main__":
    unittest.main()
