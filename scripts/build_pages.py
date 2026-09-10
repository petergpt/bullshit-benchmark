#!/usr/bin/env python3
"""Build the public Pages tree from an allowlist and verified release manifests.

This is a static packaging step, not a benchmark run or a publisher. CI runs the
semantic publication/config validators and tests before invoking it. Never copy
the repository wholesale: local runs and private artifacts can coexist with it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import sys
import tempfile
from urllib.parse import unquote, urlsplit

import publication
import published_dataset as storage

ROOT = Path(__file__).resolve().parents[1]
MAX_SITE_BYTES = 1_000_000_000
DATASETS = ('data/latest', 'data/v2/latest')
RUNTIME_FILES = (
    'index.html', 'questions.json', 'questions.v2.json', 'LICENSE',
    'viewer/index.html', 'viewer/index.v2.html', 'viewer/index.next.html', 'viewer/index.legacy.html',
    'viewer/next/app.mjs', 'viewer/next/app.css', 'viewer/next/data.mjs',
    'viewer/next/charts.mjs', 'viewer/next/labels.mjs',
    'viewer/next/brands.mjs', 'viewer/next/capture.mjs',
    'docs/images/bsbench.png',
)
OPTIONAL_FILES = (
    'CNAME', 'robots.txt', 'README.md', 'CHANGELOG.md',
    'docs/TECHNICAL.md', 'docs/STORAGE_AUDIT.md', 'viewer/next/README.md',
    'viewer/next/assets/brands/SOURCES.md',
    'viewer/next/assets/brands/LICENSE-simple-icons.md',
    'viewer/next/assets/brands/LICENSE-lobehub.txt',
    'viewer/next/assets/brands/provenance.json',
    *('data/model_metadata/' + name + '.csv' for name in (
        'model_launch_dates', 'model_params', 'model_buckets',
        'tested_models_inventory', 'model_launch_sources',
        'model_launch_collection', 'model_launch_judged', 'model_launch_attempts',
        'model_launch_dates_review', 'model_launch_dates_candidates',
    )),
)
PUBLIC_IMAGE_DIRS = ('docs/images', 'viewer/next/assets/brands')
IMAGE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|jpg|jpeg|gif|svg|webp|ico)\Z')
METADATA_NAMES = frozenset(publication.SIDECARS) | {'questions.json'}


class BuildError(ValueError):
    pass


@dataclass(frozen=True)
class PublicFile:
    source: Path
    size: int
    digest: str


def safe_file(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or ':' in relative or '%' in relative
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise BuildError(f'Unsafe public path: {relative!r}')
    path = root
    for part in relative.split('/'):
        path /= part
        if path.is_symlink():
            raise BuildError(f'Symlink public path: {relative}')
    if not path.resolve().is_relative_to(root):
        raise BuildError(f'Public path escapes source: {relative}')
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise BuildError(f'Missing public file: {relative}')
    return path


def fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def add_file(plan, root, relative, descriptor=None, *, target=None):
    source = safe_file(root, relative)
    size, digest = fingerprint(source)
    if descriptor is not None and (size != descriptor.get('bytes') or digest != descriptor.get('sha256')):
        raise BuildError(f'Public file checksum mismatch: {relative}')
    public = PublicFile(source, size, digest)
    destination = target or relative
    if destination in plan and plan[destination] != public:
        raise BuildError(f'Conflicting public file: {destination}')
    plan[destination] = public


def add_release(plan, root, dataset, manifest, *, current):
    spec = storage._storage(manifest)
    if spec is None or spec['version'] != 2:
        raise BuildError(f'Pages requires immutable storage version 2: {dataset}')
    for asset in spec['assets'].values():
        for part in asset['parts']:
            add_file(plan, root, f'{dataset}/{part["path"]}', part)
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) - METADATA_NAMES:
        raise BuildError(f'Unknown public metadata files: {dataset}')
    if current and not METADATA_NAMES.issubset(files):
        raise BuildError(f'Incomplete public metadata/question snapshot: {dataset}')
    for name, descriptor in files.items():
        expected = f'metadata/{descriptor.get("sha256")}-{name}'
        if descriptor.get('path') != expected:
            raise BuildError(f'Non-immutable public metadata: {dataset}/{name}')
        source = f'{dataset}/{expected}'
        add_file(plan, root, source, descriptor)
        if current:
            # Stable downloads must match this release, even if a local alias
            # was edited or interrupted. Copy the checked immutable bytes.
            add_file(plan, root, source, descriptor, target=f'{dataset}/{name}')


def public_files(root: Path) -> dict[str, PublicFile]:
    root = root.resolve()
    plan = {}
    for relative in RUNTIME_FILES:
        add_file(plan, root, relative)
    for relative in OPTIONAL_FILES:
        if (root / relative).exists() or (root / relative).is_symlink():
            add_file(plan, root, relative)
    for relative in PUBLIC_IMAGE_DIRS:
        directory = root / relative
        if directory.is_symlink():
            raise BuildError(f'Symlink public directory: {relative}')
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if IMAGE_NAME.fullmatch(path.name):
                    add_file(plan, root, f'{relative}/{path.name}')
    for dataset in DATASETS:
        relative = f'{dataset}/manifest.json'
        add_file(plan, root, relative)
        manifest = publication.decode(plan[relative].source.read_bytes())
        add_release(plan, root, dataset, manifest, current=True)
        previous = manifest.get('previous_manifest')
        if previous:
            expected = f'releases/{previous.get("sha256")}.json'
            if previous.get('path') != expected:
                raise BuildError(f'Invalid previous release path: {dataset}')
            relative = f'{dataset}/{expected}'
            add_file(plan, root, relative, previous)
            prior = publication.decode(plan[relative].source.read_bytes())
            add_release(plan, root, dataset, prior, current=False)
            # Retention is exactly current + previous. The prior manifest may
            # contain its own historical pointer; do not revive pruned releases.
        for group in ('sources', 'exports'):
            for value in manifest.get(group, {}).values():
                for reference in value if isinstance(value, list) else [value]:
                    if not isinstance(reference, str) or reference not in plan:
                        raise BuildError(f'Missing public {group} reference: {reference!r}')
    return plan


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.links.extend(value for key, value in attrs if key in ('src', 'href') and value)
        if tag == 'meta' and values.get('http-equiv', '').lower() == 'refresh':
            match = re.search(r'url\s*=\s*(.+)', values.get('content', ''), re.I)
            if match:
                self.links.append(match[1].strip(' \"\''))


def local_target(source: str, reference: str) -> str | None:
    url = urlsplit(reference)
    if url.scheme or url.netloc or not url.path or '${' in reference:
        return None
    path = unquote(url.path)
    if path.startswith('/') or '\\' in path:
        raise BuildError(f'Non-portable local URL in {source}: {reference}')
    target = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if target == '..' or target.startswith('../'):
        raise BuildError(f'Local URL escapes site in {source}: {reference}')
    return target


def verify_runtime_links(site: Path) -> None:
    """Check literal local HTML/CSS/module URLs, including cache-busted imports.

    Manifest URLs are checked when planning; computed response/brand URLs are
    covered by manifest descriptors, the image directory allowlist and JS tests.
    This deliberately is not a JavaScript parser or an external link crawler.
    """
    for source in sorted(site.rglob('*')):
        if source.suffix not in ('.html', '.css', '.mjs', '.js'):
            continue
        relative = source.relative_to(site).as_posix()
        content = source.read_text(encoding='utf-8')
        links = []
        if source.suffix == '.html':
            parser = HTMLLinks()
            parser.feed(content)
            links.extend(parser.links)
        if source.suffix in ('.css', '.html'):
            links.extend(re.findall(r'url\(\s*[\"\']?([^\s\"\')]+)', content))
        if source.suffix in ('.mjs', '.js', '.html'):
            links.extend(re.findall(r'(?:\bfrom\s+|\bimport\s*(?:\(\s*)?)[\"\']([^\"\']+)[\"\']', content))
            links.extend(re.findall(r'new\s+URL\(\s*[\"\']([^\"\']+)[\"\']\s*,\s*import\.meta\.url', content))
        for link in links:
            target = local_target(relative, link)
            if target is None:
                continue
            path = site / target
            if path.is_dir():
                path /= 'index.html'
            if not path.is_file():
                raise BuildError(f'Missing local URL in {relative}: {link}')


def build_pages(root: Path, output: Path, *, max_bytes=MAX_SITE_BYTES) -> dict:
    root = root.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise BuildError(f'Output must be a new directory: {output}')
    if root.is_relative_to(output.resolve()):
        raise BuildError('Output cannot replace the source or an ancestor')
    if not 0 < max_bytes <= MAX_SITE_BYTES:
        raise BuildError('Site byte limit must be positive and at most 1 GB')
    plan = public_files(root)
    size = sum(item.size for item in plan.values())
    if size >= max_bytes:
        raise BuildError(f'Pages artifact is {size:,} bytes; limit is below {max_bytes:,}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pages-build-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'site'
        stage.mkdir()
        for relative, item in sorted(plan.items()):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A copy (not a symlink or hard link) also meets Pages tar constraints.
            safe_file(root, item.source.relative_to(root).as_posix())
            shutil.copyfile(item.source, destination)
            if fingerprint(destination) != (item.size, item.digest):
                raise BuildError(f'Public source changed during build: {relative}')
        (stage / '.nojekyll').touch()
        verify_runtime_links(stage)
        os.replace(stage, output)
    return {'files': len(plan) + 1, 'bytes': size, 'max_bytes': max_bytes,
            'datasets': list(DATASETS), 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='new destination directory')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_pages(args.source, args.output), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'build_pages: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
