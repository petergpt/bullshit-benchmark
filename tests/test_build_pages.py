"""Public deployment boundaries; tiny synthetic fixtures, no API calls."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_pages as pages
import published_dataset as storage


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class PagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'repo'
        self.output = self.base / 'site'
        self.release_number = 0
        for relative in pages.RUNTIME_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        (self.root / 'index.html').write_text('<a href="viewer/index.v2.html">Viewer</a>')
        (self.root / 'viewer/index.next.html').write_text(
            '<link href="./next/app.css?v=1"><script src="./next/app.mjs?v=1"></script>'
            '<img src="../docs/images/bsbench.png">')
        for dataset in pages.DATASETS:
            self.release(dataset, 'current')

    def release(self, dataset, label):
        self.release_number += 1
        stage = self.base / f'release-{self.release_number}'
        stage.mkdir()
        row = {'sample_id': label, 'question_id': 'q', 'model': 'example/model', 'response_text': label}
        for logical, kind in storage.ASSET_FORMATS.items():
            raw = json.dumps([row]).encode() if kind == 'json-gzip' else (json.dumps(row) + '\n').encode()
            (stage / logical).write_bytes(gzip.compress(raw) if kind == 'json-gzip' else raw)
        (stage / 'manifest.json').write_text('{}')
        storage.pack_dataset(stage)
        manifest = json.loads((stage / 'manifest.json').read_bytes())
        manifest['files'] = {}
        for name in pages.METADATA_NAMES:
            raw = (json.dumps({'release': label}) + '\n').encode()
            path = f'metadata/{digest(raw)}-{name}'
            (stage / path).parent.mkdir(exist_ok=True)
            (stage / path).write_bytes(raw)
            (stage / name).write_bytes(raw)
            manifest['files'][name] = {'path': path, 'bytes': len(raw), 'sha256': digest(raw)}
        manifest['sources'] = {
            storage.SOURCE_KEYS[logical] + 's': [f'{dataset}/{part["path"]}' for part in asset['parts']]
            for logical, asset in manifest['storage']['assets'].items()
        }
        manifest['exports'] = {'leaderboard_csv': f'{dataset}/leaderboard.csv'}
        (stage / 'manifest.json').write_text(json.dumps(manifest))
        shutil.copytree(stage, self.root / dataset, dirs_exist_ok=True)
        return manifest

    def save_manifest(self, dataset, manifest):
        (self.root / dataset / 'manifest.json').write_text(json.dumps(manifest))

    def test_only_public_files_and_manifest_assets_are_copied(self):
        private_paths = (
            '.git/config', '.env', 'config.json', 'runs/secret/responses.jsonl',
            'outputs/report.html', 'reports/report.html', 'tmp/private.json',
            'data/ad_hoc/secret.json', 'viewer/next/secret.json',
            'viewer/next/assets/brands/notes.txt', 'docs/images/.secret.png',
            'data/latest/responses/sha256-' + '0' * 64 + '.jsonl',
        )
        for relative in private_paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('not public')
        logo = self.root / 'viewer/next/assets/brands/example.svg'
        logo.write_text('<svg/>')
        result = pages.build_pages(self.root, self.output)
        for relative in private_paths:
            self.assertFalse((self.output / relative).exists(), relative)
        self.assertTrue((self.output / '.nojekyll').is_file())
        self.assertTrue((self.output / logo.relative_to(self.root)).is_file())
        self.assertEqual((self.output / 'index.html').read_bytes(), (self.root / 'index.html').read_bytes())
        self.assertEqual(result['bytes'], sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()))
        self.assertTrue(all(not p.is_symlink() and p.stat().st_nlink == 1 for p in self.output.rglob('*') if p.is_file()))

    def test_stable_exports_are_copied_from_the_validated_snapshot(self):
        source = self.root / 'data/latest'
        (source / 'leaderboard.csv').write_text('stale CSV')
        (source / 'questions.json').write_text('stale questions')
        manifest = json.loads((source / 'manifest.json').read_bytes())
        pages.build_pages(self.root, self.output)
        for name in ('leaderboard.csv', 'questions.json'):
            expected = (source / manifest['files'][name]['path']).read_bytes()
            self.assertEqual((self.output / 'data/latest' / name).read_bytes(), expected)

    def test_retains_exactly_current_and_previous_declared_assets(self):
        dataset = 'data/latest'
        previous = json.loads((self.root / dataset / 'manifest.json').read_bytes())
        previous['previous_manifest'] = {'path': 'releases/' + '0' * 64 + '.json'}
        previous_raw = json.dumps(previous).encode()
        previous_path = f'releases/{digest(previous_raw)}.json'
        current = self.release(dataset, 'new')
        current['previous_manifest'] = {'path': previous_path, 'bytes': len(previous_raw), 'sha256': digest(previous_raw)}
        path = self.root / dataset / previous_path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(previous_raw)
        self.save_manifest(dataset, current)
        pages.build_pages(self.root, self.output)
        self.assertEqual((self.output / dataset / previous_path).read_bytes(), previous_raw)
        for release in (current, previous):
            for relative in publication_references(release):
                self.assertTrue((self.output / dataset / relative).is_file(), relative)
        self.assertFalse((self.output / dataset / previous['previous_manifest']['path']).exists())

    def test_missing_or_modified_part_fails_before_creating_artifact(self):
        dataset = self.root / 'data/latest'
        manifest = json.loads((dataset / 'manifest.json').read_bytes())
        part = dataset / manifest['storage']['assets']['responses.jsonl']['parts'][0]['path']
        part.write_bytes(b'corrupt')
        with self.assertRaisesRegex(pages.BuildError, 'checksum'):
            pages.build_pages(self.root, self.output)
        self.assertFalse(self.output.exists())
        part.unlink()
        with self.assertRaisesRegex(pages.BuildError, 'Missing public file'):
            pages.build_pages(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_metadata_cannot_add_private_or_traversing_paths(self):
        dataset = 'data/latest'
        manifest = json.loads((self.root / dataset / 'manifest.json').read_bytes())
        manifest['files']['questions.json']['path'] = '../../private.json'
        self.save_manifest(dataset, manifest)
        with self.assertRaisesRegex(pages.BuildError, 'Non-immutable'):
            pages.build_pages(self.root, self.output)
        manifest['files']['private.json'] = manifest['files'].pop('questions.json')
        self.save_manifest(dataset, manifest)
        with self.assertRaisesRegex(pages.BuildError, 'Unknown public metadata'):
            pages.build_pages(self.root, self.output)

    def test_public_symlinks_are_rejected(self):
        secret = self.base / 'private.txt'
        secret.write_text('private')
        logo = self.root / 'docs/images/bsbench.png'
        logo.unlink()
        logo.symlink_to(secret)
        with self.assertRaisesRegex(pages.BuildError, 'Symlink'):
            pages.build_pages(self.root, self.output)

    def test_frozen_questions_and_export_references_are_required(self):
        dataset = 'data/latest'
        manifest = json.loads((self.root / dataset / 'manifest.json').read_bytes())
        questions = manifest['files'].pop('questions.json')
        self.save_manifest(dataset, manifest)
        with self.assertRaisesRegex(pages.BuildError, 'question snapshot'):
            pages.build_pages(self.root, self.output)
        manifest['files']['questions.json'] = questions
        manifest['exports']['private'] = 'outputs/private.json'
        self.save_manifest(dataset, manifest)
        with self.assertRaisesRegex(pages.BuildError, 'Missing public exports reference'):
            pages.build_pages(self.root, self.output)

    def test_missing_cache_busted_module_fails_atomically(self):
        (self.root / 'viewer/next/app.mjs').write_text('import "./missing.mjs?v=123";')
        with self.assertRaisesRegex(pages.BuildError, 'Missing local URL'):
            pages.build_pages(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_site_size_limit_is_strict(self):
        size = sum(item.size for item in pages.public_files(self.root).values())
        with self.assertRaisesRegex(pages.BuildError, 'limit'):
            pages.build_pages(self.root, self.output, max_bytes=size)
        self.assertFalse(self.output.exists())

    def test_existing_destination_is_never_deleted(self):
        self.output.mkdir()
        keep = self.output / 'keep.txt'
        keep.write_text('existing work')
        with self.assertRaisesRegex(pages.BuildError, 'new directory'):
            pages.build_pages(self.root, self.output)
        self.assertEqual(keep.read_text(), 'existing work')

    def test_source_changed_during_copy_is_rejected(self):
        original_copy = shutil.copyfile
        def replace_copy(source, destination):
            if source.name == 'index.html' and source.parent == self.root:
                source.write_text('changed after planning')
            return original_copy(source, destination)
        with patch.object(pages.shutil, 'copyfile', side_effect=replace_copy):
            with self.assertRaisesRegex(pages.BuildError, 'changed during build'):
                pages.build_pages(self.root, self.output)
        self.assertFalse(self.output.exists())


def publication_references(manifest):
    return [part['path'] for asset in manifest['storage']['assets'].values() for part in asset['parts']] + [entry['path'] for entry in manifest['files'].values()]


if __name__ == '__main__':
    unittest.main()
