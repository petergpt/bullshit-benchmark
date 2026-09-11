#!/usr/bin/env python3
"""Offline validation of reusable configs and published model coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import openrouter_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ("config.json", "config.v2.json", "config.new-models.v1.json", "config.new-models.v2.json")
CANDIDATE_CONFIG_NAMES = frozenset(("config.new-models.v1.json", "config.new-models.v2.json"))
SUITES = {"v1": ("config.json", "data/latest"), "v2": ("config.v2.json", "data/v2/latest")}
EXCEPTIONS_PATH = "data/model_metadata/legacy_config_exceptions.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _matches_model(key, models):
    return key == "*" or key in models or (key.endswith("/*") and any(model.startswith(key[:-1]) for model in models))


def configured_variants(config_path: Path, *, allow_empty: bool = False) -> set[str]:
    """Validate a config and return the exact labels produced by the collector."""
    config_path = Path(config_path)
    config = read_json(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("collect"), dict):
        raise ValueError(f"{config_path}: collect must be an object")
    collect = config["collect"]
    models = collect.get("models")
    if not isinstance(models, list) or (not models and not allow_empty) or any(not isinstance(model, str) or not model.strip() or model != model.strip() for model in models):
        raise ValueError(f"{config_path}: collect.models must contain non-empty model IDs")
    if len(set(models)) != len(models):
        raise ValueError(f"{config_path}: duplicate collect.models entries")
    if collect.get("models_file"):
        raise ValueError(f"{config_path}: durable configs must declare models inline so publication coverage is reviewable")
    for name in ("model_reasoning_efforts", "model_request_overrides", "model_providers"):
        mapping = collect.get(name, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"{config_path}: collect.{name} must be an object")
        for key, value in mapping.items():
            if key != key.strip() or not (key in models if name == "model_reasoning_efforts" else _matches_model(key, models)):
                raise ValueError(f"{config_path}: collect.{name} key {key!r} matches no configured model")
            if name == "model_reasoning_efforts":
                if not isinstance(value, list) or any(not isinstance(effort, str) or not effort.strip() for effort in value):
                    raise ValueError(f"{config_path}: reasoning efforts for {key} must be an array of strings")
                normalized = [benchmark.normalize_reasoning_effort(effort, field_name=f"{config_path}: {key}") for effort in value]
                if len(set(normalized)) != len(normalized):
                    raise ValueError(f"{config_path}: duplicate reasoning efforts for {key}")
    try:
        default_effort = benchmark.normalize_reasoning_effort(collect.get("response_reasoning_effort", "off"), field_name="response_reasoning_effort")
        efforts = benchmark.parse_model_reasoning_efforts(collect.get("model_reasoning_efforts", {}))
        providers = benchmark.parse_model_providers(collect.get("model_providers", {}), field_name="model_providers")
        overrides = benchmark.parse_model_request_overrides(collect.get("model_request_overrides", {}), field_name="model_request_overrides")
        for model in models:
            override = benchmark.resolve_model_request_overrides(model, overrides).get("reasoning", {})
            if isinstance(override, dict) and "effort" in override:
                effort = benchmark.normalize_reasoning_effort(override["effort"], field_name=f"reasoning override for {model}")
                declared = efforts.get(model, [default_effort]) or [None]
                if effort not in declared:
                    raise ValueError(f"reasoning override for {model} is absent from model_reasoning_efforts")
        variants = benchmark.build_model_variants(models, default_effort, efforts, providers, overrides)
    except ValueError as exc:
        raise ValueError(f"{config_path}: {exc}") from exc
    labels = [variant["model_label"] for variant in variants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{config_path}: multiple entries resolve to the same model/reasoning variant")
    return set(labels)


def published_variants(dataset_dir: Path) -> set[str]:
    path = dataset_dir / "leaderboard.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "model" not in reader.fieldnames:
                raise ValueError("missing model column")
            variants = []
            for number, row in enumerate(reader, start=2):
                model = row.get("model")
                if not isinstance(model, str) or not model.strip() or model != model.strip() or None in row:
                    raise ValueError(f"invalid model row at line {number}")
                variants.append(model)
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("model roster is empty or contains duplicate variants")
        return set(variants)
    except (OSError, ValueError, csv.Error) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def validate_repository(root: Path = ROOT) -> dict:
    root = Path(root)
    configured = {
        name: configured_variants(root / name, allow_empty=name in CANDIDATE_CONFIG_NAMES)
        for name in CONFIG_NAMES
    }
    path = root / EXCEPTIONS_PATH
    exceptions = read_json(path)
    if not isinstance(exceptions, dict) or exceptions.get("schema_version") != 1 or not isinstance(exceptions.get("reason"), str) or not exceptions["reason"].strip():
        raise ValueError(f"{path}: expected schema_version 1 and a non-empty historical exception reason")
    suites = exceptions.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(SUITES):
        raise ValueError(f"{path}: suites must contain exactly v1 and v2")
    result = {"configs": {name: len(variants) for name, variants in configured.items()}, "suites": {}}
    for suite, (config_name, dataset_path) in SUITES.items():
        legacy = suites[suite]
        if not isinstance(legacy, list) or any(not isinstance(value, str) or not value.strip() or "*" in value for value in legacy) or len(set(legacy)) != len(legacy):
            raise ValueError(f"{path}: {suite} exceptions must be unique, exact model/reasoning strings")
        published = published_variants(root / dataset_path)
        unconfigured = published - configured[config_name]
        missing = unconfigured - set(legacy)
        stale = set(legacy) - unconfigured
        if missing:
            raise ValueError(f"{suite}: published variants absent from {config_name} and historical exceptions: {', '.join(sorted(missing))}")
        if stale:
            raise ValueError(f"{suite}: remove stale historical exceptions now configured or unpublished: {', '.join(sorted(stale))}")
        result["suites"][suite] = {"published_variants": len(published), "historical_exceptions": len(legacy)}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate_repository(args.root), indent=2, sort_keys=True))
    except ValueError as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
