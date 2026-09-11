#!/usr/bin/env python3
"""Check OpenRouter for text-output models missing from BullshitBench coverage."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_CATALOG_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CONFIG_GLOB = "config*.json"
DEFAULT_LEADERBOARD_PATHS = (
    "data/latest/leaderboard.csv",
    "data/v2/latest/leaderboard.csv",
)
DEFAULT_METADATA_PATHS = ("data/model_metadata/tested_models_inventory.csv",)
DEFAULT_ALIAS_PATH = "data/model_metadata/openrouter_model_aliases.json"
DEFAULT_RECENT_DAYS = 21
DEFAULT_EXCLUDED_MODELS = {
    "anthropic/claude-opus-4.7-fast",
    "anthropic/claude-opus-4.8-fast",
    "inclusionai/ring-2.6-1t",
    "cohere/north-mini-code:free",
    "google/gemini-3-pro-image",
    "google/gemini-3.1-flash-image",
    "moonshotai/kimi-k2.7-code",
    "nex-agi/nex-n2-pro",
    "nex-agi/nex-n2-pro:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "openai/gpt-5.4-pro",
    "openrouter/auto",
    "openrouter/fusion",
    "perceptron/perceptron-mk1",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs.2:free",
    "qwen/qwen3.7-plus",
    "stepfun/step-3.7-flash",
    "x-ai/grok-build-0.1",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_from_unix(value: Any) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return ""
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def strip_reasoning_suffix(model_id: str) -> str:
    return str(model_id or "").split("@reasoning=", 1)[0].strip()


def normalized_model_id(model_id: str) -> str:
    return strip_reasoning_suffix(model_id).lower()


def alias_forms(model_id: str) -> set[str]:
    model_id = normalized_model_id(model_id)
    if not model_id:
        return set()
    forms = {model_id}
    if model_id.endswith(":free"):
        forms.add(model_id.removesuffix(":free"))
    return forms


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_catalog(args: argparse.Namespace) -> dict[str, Any]:
    if args.catalog_json:
        return read_json(pathlib.Path(args.catalog_json))

    request = urllib.request.Request(
        args.catalog_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BullshitBench OpenRouter model check",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to fetch OpenRouter catalog: {exc}") from exc


def iter_config_paths(patterns: list[str]) -> list[pathlib.Path]:
    paths: set[pathlib.Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = pathlib.Path(match)
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def collect_config_models(path: pathlib.Path) -> set[str]:
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"Could not read config {path}: {exc}") from exc
    collect = data.get("collect", {}) if isinstance(data, dict) else {}
    models = collect.get("models", []) if isinstance(collect, dict) else []
    if not isinstance(models, list):
        return set()
    return {normalized_model_id(model) for model in models if str(model).strip()}


def collect_leaderboard_models(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    models: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model") or row.get("model_id") or ""
            if model:
                models.add(normalized_model_id(model))
    return models


def collect_metadata_models(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    models: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model_id") or row.get("model") or ""
            if model:
                models.add(normalized_model_id(model))
    return models


def load_alias_map(path: pathlib.Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"Alias file {path} must be a JSON object.")
    aliases: dict[str, list[str]] = {}
    for source, equivalent in data.items():
        source_id = normalized_model_id(str(source))
        if not source_id:
            continue
        if isinstance(equivalent, str):
            equivalents = [equivalent]
        elif isinstance(equivalent, list):
            equivalents = [str(item) for item in equivalent]
        else:
            raise SystemExit(
                f"Alias value for {source!r} must be a string or list of strings."
            )
        aliases[source_id] = [normalized_model_id(item) for item in equivalents if item]
    return aliases


def is_text_model(model: dict[str, Any]) -> bool:
    architecture = model.get("architecture") or {}
    input_modalities = set(architecture.get("input_modalities") or [])
    output_modalities = set(architecture.get("output_modalities") or [])
    if input_modalities or output_modalities:
        return "text" in input_modalities and "text" in output_modalities
    modality = str(architecture.get("modality") or "").lower()
    return "text" in modality and "->text" in modality


def model_aliases(model: dict[str, Any]) -> set[str]:
    aliases = set()
    for field in ("id", "canonical_slug"):
        value = model.get(field)
        if value:
            aliases.update(alias_forms(str(value)))
    return aliases


def known_equivalent(aliases: set[str], known_models: set[str], alias_map: dict[str, list[str]]) -> str:
    for alias in sorted(aliases):
        if alias in known_models:
            return alias
        for equivalent in alias_map.get(alias, []):
            if equivalent in known_models:
                return equivalent
    return ""


def excluded_by_default(aliases: set[str], excluded_models: set[str]) -> bool:
    return any(alias.startswith("~") or alias in excluded_models for alias in aliases)


def sorted_candidates(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        models,
        key=lambda model: (
            int(model.get("created") or 0),
            str(model.get("id") or ""),
        ),
        reverse=True,
    )


def candidate_record(model: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    architecture = model.get("architecture") or {}
    pricing = model.get("pricing") or {}
    top_provider = model.get("top_provider") or {}
    created = model.get("created")
    try:
        created_dt = dt.datetime.fromtimestamp(int(created), tz=dt.timezone.utc)
        age_days = (now - created_dt).days
    except (TypeError, ValueError, OSError):
        age_days = None
    supported_parameters = sorted(str(item) for item in model.get("supported_parameters") or [])
    return {
        "id": model.get("id"),
        "canonical_slug": model.get("canonical_slug"),
        "name": model.get("name"),
        "created_utc": iso_from_unix(created),
        "age_days": age_days,
        "context_length": model.get("context_length"),
        "max_completion_tokens": top_provider.get("max_completion_tokens"),
        "input_modalities": architecture.get("input_modalities") or [],
        "output_modalities": architecture.get("output_modalities") or [],
        "modality": architecture.get("modality"),
        "prompt_price": pricing.get("prompt"),
        "completion_price": pricing.get("completion"),
        "reasoning_supported": "reasoning" in supported_parameters,
        "include_reasoning_supported": "include_reasoning" in supported_parameters,
        "supported_parameters": supported_parameters,
    }


def render_markdown(result: dict[str, Any], max_rows: int) -> str:
    lines = [
        "# OpenRouter New Text Model Check",
        "",
        f"- Checked at: `{result['checked_at_utc']}`",
        f"- Catalog URL: `{result['catalog_url']}`",
        f"- Known model aliases compared: `{result['known_model_alias_count']}`",
        f"- Text-capable catalog models: `{result['text_model_count']}`",
        f"- Recent window: `last {result['recent_days']} days`" if result["recent_days"] else "- Recent window: `disabled`",
        f"- Suppressed older missing models: `{result['suppressed_older_candidate_count']}`",
        f"- Suppressed excluded/non-test targets: `{result['excluded_model_count']}`",
        f"- Candidate models: `{result['candidate_count']}`",
        "",
    ]
    if result["candidate_count"] == 0:
        lines.append("No new text-output OpenRouter models were found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Candidates",
            "",
            "| Created | Model | Name | Modalities | Context | Pricing prompt/completion | Reasoning |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for candidate in result["candidates"][:max_rows]:
        modalities = candidate.get("modality") or (
            f"{'+'.join(candidate.get('input_modalities') or [])}->"
            f"{'+'.join(candidate.get('output_modalities') or [])}"
        )
        context_length = candidate.get("context_length")
        context = f"{context_length:,}" if isinstance(context_length, int) else str(context_length or "")
        pricing = f"{candidate.get('prompt_price') or '?'} / {candidate.get('completion_price') or '?'}"
        reasoning = "yes" if candidate.get("reasoning_supported") else "no"
        lines.append(
            "| {created} | `{model_id}` | {name} | `{modalities}` | {context} | `{pricing}` | {reasoning} |".format(
                created=candidate.get("created_utc") or "",
                model_id=candidate.get("id") or "",
                name=str(candidate.get("name") or "").replace("|", "\\|"),
                modalities=str(modalities).replace("|", "\\|"),
                context=context,
                pricing=pricing,
                reasoning=reasoning,
            )
        )
    if result["candidate_count"] > max_rows:
        lines.extend(["", f"_Showing first {max_rows} candidates._"])
    lines.extend(
        [
            "",
            "Suggested next step: inspect the top candidates, probe reasoning support once, then add the chosen models to `config.new-models.v1.json` and `config.new-models.v2.json` for a catch-up sweep.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find OpenRouter text-output models missing from local configs and published leaderboards."
    )
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    parser.add_argument("--catalog-json", default="", help="Read a saved catalog JSON instead of fetching OpenRouter.")
    parser.add_argument(
        "--config-glob",
        action="append",
        default=[],
        help=f"Config glob to scan. Default: {DEFAULT_CONFIG_GLOB}",
    )
    parser.add_argument(
        "--leaderboard",
        action="append",
        default=[],
        help="Leaderboard CSV to scan. Defaults to published v1/v2 paths.",
    )
    parser.add_argument(
        "--metadata-csv",
        action="append",
        default=[],
        help="Metadata CSV with model_id/model column to scan.",
    )
    parser.add_argument("--alias-json", default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--max-markdown-rows", type=int, default=40)
    parser.add_argument(
        "--recent-days",
        type=int,
        default=DEFAULT_RECENT_DAYS,
        help="Only report missing models created in the last N days. Use 0 to include older candidates.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Additional model ID to suppress from candidate output.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--fail-on-candidates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = utc_now()
    config_patterns = args.config_glob or [DEFAULT_CONFIG_GLOB]
    leaderboard_paths = [pathlib.Path(path) for path in (args.leaderboard or DEFAULT_LEADERBOARD_PATHS)]
    metadata_paths = [pathlib.Path(path) for path in (args.metadata_csv or DEFAULT_METADATA_PATHS)]
    alias_map = load_alias_map(pathlib.Path(args.alias_json))

    known_models: set[str] = set()
    config_paths = iter_config_paths(config_patterns)
    for path in config_paths:
        known_models.update(collect_config_models(path))
    for path in leaderboard_paths:
        known_models.update(collect_leaderboard_models(path))
    for path in metadata_paths:
        known_models.update(collect_metadata_models(path))

    expanded_known: set[str] = set()
    for model_id in known_models:
        expanded_known.update(alias_forms(model_id))
    known_models = expanded_known

    catalog = read_catalog(args)
    raw_models = catalog.get("data") if isinstance(catalog, dict) else None
    if not isinstance(raw_models, list):
        raise SystemExit("OpenRouter catalog JSON did not contain a data array.")

    text_models = [model for model in raw_models if isinstance(model, dict) and is_text_model(model)]
    candidates = []
    suppressed_older_count = 0
    excluded_count = 0
    excluded_models = set(DEFAULT_EXCLUDED_MODELS)
    excluded_models.update(normalized_model_id(model) for model in args.exclude_model if model)
    for model in sorted_candidates(text_models):
        aliases = model_aliases(model)
        if excluded_by_default(aliases, excluded_models):
            excluded_count += 1
            continue
        equivalent = known_equivalent(aliases, known_models, alias_map)
        if equivalent:
            continue
        record = candidate_record(model, now)
        age_days = record.get("age_days")
        if args.recent_days > 0 and isinstance(age_days, int) and age_days > args.recent_days:
            suppressed_older_count += 1
            continue
        candidates.append(record)

    result = {
        "checked_at_utc": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "catalog_url": args.catalog_url,
        "config_paths": [str(path) for path in config_paths],
        "leaderboard_paths": [str(path) for path in leaderboard_paths if path.exists()],
        "metadata_paths": [str(path) for path in metadata_paths if path.exists()],
        "alias_path": args.alias_json if pathlib.Path(args.alias_json).exists() else "",
        "catalog_model_count": len(raw_models),
        "text_model_count": len(text_models),
        "recent_days": args.recent_days,
        "known_model_alias_count": len(known_models),
        "excluded_model_count": excluded_count,
        "suppressed_older_candidate_count": suppressed_older_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    rendered = render_markdown(result, max_rows=args.max_markdown_rows)
    if args.output_json:
        pathlib.Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        pathlib.Path(args.output_md).write_text(rendered, encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(rendered, end="")

    if candidates and args.fail_on_candidates:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
