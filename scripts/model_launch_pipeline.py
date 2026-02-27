#!/usr/bin/env python3
"""Model launch-date provenance pipeline.

This script builds model inventory, prepares provider-bucket collector tasks,
judges launch-date evidence, and exports review/candidate/canonical datasets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import re
from typing import Any
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "model_metadata"
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_LATEST_AGGREGATE = ROOT / "data" / "latest" / "aggregate.jsonl"
DEFAULT_LATEST_RESPONSES = ROOT / "data" / "latest" / "responses.jsonl"
DEFAULT_RUNS_DIR = ROOT / "runs"

INVENTORY_CSV = DATA_DIR / "tested_models_inventory.csv"
BUCKETS_CSV = DATA_DIR / "model_buckets.csv"
SOURCES_CSV = DATA_DIR / "model_launch_sources.csv"
COLLECTION_CSV = DATA_DIR / "model_launch_collection.csv"
JUDGED_CSV = DATA_DIR / "model_launch_judged.csv"
ATTEMPTS_CSV = DATA_DIR / "model_launch_attempts.csv"
REVIEW_CSV = DATA_DIR / "model_launch_dates_review.csv"
CANDIDATES_CSV = DATA_DIR / "model_launch_dates_candidates.csv"
CANONICAL_CSV = DATA_DIR / "model_launch_dates.csv"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

COLLECTOR_FIELDS = [
    "model_id",
    "org",
    "proposed_launch_date",
    "evidence_url",
    "evidence_domain",
    "evidence_title",
    "evidence_published_date",
    "evidence_type",
    "supporting_snippet",
    "notes",
    "collector_agent",
    "collected_at_utc",
]

JUDGE_FIELDS = COLLECTOR_FIELDS + [
    "judge_status",
    "judge_reason",
    "required_fix",
    "judge_confidence",
    "judged_at_utc",
    "attempt_count",
]

REVIEW_FIELDS = [
    "model_id",
    "org",
    "launch_date",
    "evidence_url",
    "evidence_title",
    "evidence_published_date",
    "judge_status",
    "judge_reason",
    "attempt_count",
    "manual_override",
    "manual_notes",
]

CANONICAL_FIELDS = [
    "model_id",
    "org",
    "launch_date",
    "evidence_url",
    "evidence_title",
    "evidence_published_date",
    "evidence_type",
    "judge_status",
    "notes",
    "updated_at_utc",
]

AGENT_BUCKETS: list[tuple[str, set[str]]] = [
    ("collector_openai", {"openai"}),
    ("collector_anthropic", {"anthropic"}),
    ("collector_google", {"google"}),
    ("collector_frontier_other", {"x-ai", "mistralai", "prime-intellect"}),
    (
        "collector_apac_longtail",
        {
            "baidu",
            "bytedance-seed",
            "deepseek",
            "minimax",
            "moonshotai",
            "qwen",
            "xiaomi",
            "z-ai",
        },
    ),
]

FIRST_PARTY_DOMAINS: dict[str, set[str]] = {
    "anthropic": {"anthropic.com"},
    "openai": {"openai.com"},
    "google": {
        "google.com",
        "blog.google",
        "deepmind.google",
        "ai.google.dev",
        "cloud.google.com",
        "developers.googleblog.com",
    },
    "x-ai": {"x.ai"},
    "mistralai": {"mistral.ai"},
    "prime-intellect": {"primeintellect.ai"},
    "baidu": {"baidu.com"},
    "bytedance-seed": {"bytedance.com"},
    "deepseek": {"deepseek.com"},
    "minimax": {"minimax.io", "minimaxi.com"},
    "moonshotai": {"moonshot.cn", "kimi.ai", "kimi.com"},
    "qwen": {"qwenlm.ai", "alibabagroup.com", "alibaba.com", "alibabacloud.com"},
    "xiaomi": {"xiaomi.com"},
    "z-ai": {"z.ai"},
}


def now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def today_utc() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_model_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"@reasoning=[^@]+$", "", text)


def derive_variant(model: str, model_reasoning_level: Any) -> str:
    raw = str(model or "").strip()
    if raw:
        return raw
    base = normalize_model_id(model)
    if not base:
        return ""
    level = str(model_reasoning_level or "default").strip() or "default"
    return f"{base}@reasoning={level}"


def parse_iso_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if not text or not DATE_RE.match(text):
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def domain_from_url(url: str) -> str:
    parsed = urlparse(url or "")
    return (parsed.netloc or "").lower().strip().removeprefix("www.")


def is_first_party_domain(org: str, domain: str) -> bool:
    allowed = FIRST_PARTY_DOMAINS.get(org, set())
    if not allowed or not domain:
        return False
    for allow in allowed:
        if domain == allow or domain.endswith(f".{allow}"):
            return True
    return False


def collector_for_org(org: str) -> str:
    for agent, orgs in AGENT_BUCKETS:
        if org in orgs:
            return agent
    return "collector_frontier_other"


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(out)


def configured_variants(config_collect: dict[str, Any], model_id: str) -> list[str]:
    raw_default = str(config_collect.get("response_reasoning_effort", "off")).strip().lower()
    default_effort = "default" if raw_default in {"", "off"} else raw_default

    per_model = config_collect.get("model_reasoning_efforts", {})
    if not isinstance(per_model, dict):
        per_model = {}

    configured = per_model.get(model_id)
    if configured is None:
        return [f"{model_id}@reasoning={default_effort}"]

    if isinstance(configured, list) and configured:
        levels = [str(level).strip().lower() for level in configured if str(level).strip()]
        return [f"{model_id}@reasoning={level}" for level in levels] or [f"{model_id}@reasoning=default"]

    return [f"{model_id}@reasoning=default"]


def add_observation(
    model_map: dict[str, dict[str, Any]],
    model_id: str,
    variant: str,
    *,
    in_config: bool = False,
    in_latest: bool = False,
    in_runs: bool = False,
) -> None:
    normalized = normalize_model_id(model_id)
    if not normalized:
        return
    org = normalized.split("/", 1)[0] if "/" in normalized else "unknown"
    entry = model_map.setdefault(
        normalized,
        {
            "model_id": normalized,
            "org": org,
            "present_in_latest": False,
            "present_in_config": False,
            "present_in_runs_history": False,
            "variants": set(),
        },
    )
    if variant:
        entry["variants"].add(variant)
    if in_config:
        entry["present_in_config"] = True
    if in_latest:
        entry["present_in_latest"] = True
    if in_runs:
        entry["present_in_runs_history"] = True


def scan_inventory(
    config_path: pathlib.Path,
    latest_aggregate_path: pathlib.Path,
    latest_responses_path: pathlib.Path,
    runs_dir: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_map: dict[str, dict[str, Any]] = {}

    if config_path.exists():
        config = read_json(config_path)
        collect_cfg = config.get("collect", {}) if isinstance(config, dict) else {}
        if isinstance(collect_cfg, dict):
            models = collect_cfg.get("models", [])
            if isinstance(models, list):
                for model in models:
                    model_id = normalize_model_id(model)
                    for variant in configured_variants(collect_cfg, model_id):
                        add_observation(
                            model_map,
                            model_id,
                            variant,
                            in_config=True,
                        )

    for path in (latest_aggregate_path, latest_responses_path):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            model = str(row.get("model", "")).strip()
            model_id = normalize_model_id(row.get("model_id") or model)
            variant = model or derive_variant(model_id, row.get("model_reasoning_level"))
            add_observation(
                model_map,
                model_id,
                variant,
                in_latest=True,
            )

    if runs_dir.exists():
        for pattern in ("**/responses.jsonl", "**/aggregate.jsonl"):
            for path in runs_dir.glob(pattern):
                for row in read_jsonl(path):
                    model = str(row.get("model", "")).strip()
                    model_id = normalize_model_id(row.get("model_id") or model)
                    variant = model or derive_variant(model_id, row.get("model_reasoning_level"))
                    add_observation(
                        model_map,
                        model_id,
                        variant,
                        in_runs=True,
                    )

        for path in runs_dir.glob("**/collection_meta.json"):
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue

            models = payload.get("models", [])
            if isinstance(models, list):
                for model in models:
                    model_id = normalize_model_id(model)
                    add_observation(
                        model_map,
                        model_id,
                        f"{model_id}@reasoning=default",
                        in_runs=True,
                    )

            variants = payload.get("model_variants", [])
            if isinstance(variants, list):
                for variant_row in variants:
                    if not isinstance(variant_row, dict):
                        continue
                    model_label = str(variant_row.get("model_label", "")).strip()
                    model_id = normalize_model_id(variant_row.get("model_id") or model_label)
                    variant = model_label or derive_variant(model_id, variant_row.get("model_reasoning_level"))
                    add_observation(
                        model_map,
                        model_id,
                        variant,
                        in_runs=True,
                    )

    inventory_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for model_id in sorted(model_map.keys()):
        entry = model_map[model_id]
        variants = sorted(str(v) for v in entry["variants"] if str(v).strip())
        org = str(entry["org"])
        collector_agent = collector_for_org(org)

        inventory_rows.append(
            {
                "model_id": model_id,
                "org": org,
                "present_in_latest": str(bool(entry["present_in_latest"])).lower(),
                "present_in_config": str(bool(entry["present_in_config"])) .lower(),
                "present_in_runs_history": str(bool(entry["present_in_runs_history"])) .lower(),
                "variant_count": len(variants),
                "variants": ";".join(variants),
            }
        )

        bucket_rows.append(
            {
                "model_id": model_id,
                "org": org,
                "collector_agent": collector_agent,
                "bucket": collector_agent.removeprefix("collector_"),
            }
        )

    return inventory_rows, bucket_rows


def build_sources_template(
    inventory_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collector_map = {row["model_id"]: row["collector_agent"] for row in bucket_rows}
    rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        model_id = row["model_id"]
        rows.append(
            {
                "model_id": model_id,
                "org": row["org"],
                "proposed_launch_date": "",
                "evidence_url": "",
                "evidence_domain": "",
                "evidence_title": "",
                "evidence_published_date": "",
                "evidence_type": "",
                "supporting_snippet": "",
                "notes": "",
                "collector_agent": collector_map.get(model_id, collector_for_org(row["org"])),
                "collected_at_utc": "",
            }
        )
    return rows


def sources_by_model(path: pathlib.Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        model_id = normalize_model_id(row.get("model_id", ""))
        if not model_id:
            continue
        out[model_id] = {k: str(v or "") for k, v in row.items()}
    return out


def collect_rows_for_models(
    model_ids: list[str],
    inventory_by_model: dict[str, dict[str, Any]],
    bucket_by_model: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, str]],
    *,
    attempt_count: int,
) -> list[dict[str, Any]]:
    collected_at = now_utc_iso()
    out: list[dict[str, Any]] = []

    for model_id in model_ids:
        inv = inventory_by_model[model_id]
        bucket = bucket_by_model[model_id]
        source = source_rows.get(model_id, {})

        evidence_url = str(source.get("evidence_url", "")).strip()
        evidence_domain = str(source.get("evidence_domain", "")).strip().lower()
        if not evidence_domain and evidence_url:
            evidence_domain = domain_from_url(evidence_url)

        notes = str(source.get("notes", "")).strip()
        if not notes and not evidence_url:
            notes = "No source row/evidence provided for this model yet."

        row: dict[str, Any] = {
            "model_id": model_id,
            "org": inv["org"],
            "proposed_launch_date": str(source.get("proposed_launch_date", "")).strip(),
            "evidence_url": evidence_url,
            "evidence_domain": evidence_domain,
            "evidence_title": str(source.get("evidence_title", "")).strip(),
            "evidence_published_date": str(source.get("evidence_published_date", "")).strip(),
            "evidence_type": str(source.get("evidence_type", "")).strip(),
            "supporting_snippet": str(source.get("supporting_snippet", "")).strip(),
            "notes": notes,
            "collector_agent": str(source.get("collector_agent", "")).strip()
            or bucket["collector_agent"],
            "collected_at_utc": str(source.get("collected_at_utc", "")).strip() or collected_at,
            "attempt_count": attempt_count,
        }
        out.append(row)

    return out


def judge_row(row: dict[str, Any], *, max_attempts: int) -> dict[str, Any]:
    judged = dict(row)
    org = str(row.get("org", "")).strip()
    model_id = str(row.get("model_id", "")).strip()

    proposed_launch_date_raw = str(row.get("proposed_launch_date", "")).strip()
    evidence_url = str(row.get("evidence_url", "")).strip()
    evidence_domain = str(row.get("evidence_domain", "")).strip().lower() or domain_from_url(evidence_url)
    evidence_title = str(row.get("evidence_title", "")).strip()
    evidence_published_raw = str(row.get("evidence_published_date", "")).strip()
    evidence_type = str(row.get("evidence_type", "")).strip()
    supporting_snippet = str(row.get("supporting_snippet", "")).strip()
    attempt_count = int(row.get("attempt_count", 1) or 1)

    hard_failures: list[str] = []
    retry_needed: list[str] = []

    proposed_launch_date = parse_iso_date(proposed_launch_date_raw)
    if not proposed_launch_date_raw:
        retry_needed.append("Missing proposed launch date.")
    elif proposed_launch_date is None:
        hard_failures.append("proposed_launch_date must be YYYY-MM-DD.")
    elif proposed_launch_date > today_utc():
        hard_failures.append("proposed_launch_date cannot be in the future.")

    if not evidence_url:
        retry_needed.append("Missing evidence_url.")

    if not evidence_domain:
        retry_needed.append("Missing evidence_domain (or parseable domain from evidence_url).")
    elif not is_first_party_domain(org, evidence_domain):
        hard_failures.append(
            f"Evidence domain '{evidence_domain}' is not first-party for org '{org}'."
        )

    if not evidence_title:
        retry_needed.append("Missing evidence_title.")

    if evidence_type not in {"announcement_blog", "newsroom", "docs_release_note", "changelog"}:
        retry_needed.append(
            "evidence_type must be one of: announcement_blog|newsroom|docs_release_note|changelog."
        )

    published_date = parse_iso_date(evidence_published_raw)
    if not evidence_published_raw:
        retry_needed.append("Missing evidence_published_date.")
    elif published_date is None:
        hard_failures.append("evidence_published_date must be YYYY-MM-DD.")
    elif published_date > today_utc():
        hard_failures.append("evidence_published_date cannot be in the future.")

    if proposed_launch_date and published_date and proposed_launch_date != published_date:
        hard_failures.append(
            "proposed_launch_date must match evidence_published_date under strict policy."
        )

    if not supporting_snippet:
        retry_needed.append("Missing supporting_snippet with key launch evidence.")

    # Basic model-specific check with normalized matching so punctuation variants
    # like "Claude 3.5 Sonnet" and "claude-3.5-sonnet" are treated as equivalent.
    model_tail = model_id.split("/", 1)[1] if "/" in model_id else model_id
    context_text = " ".join([evidence_url.lower(), evidence_title.lower(), supporting_snippet.lower()])
    if model_tail:
        tail_norm = re.sub(r"[^a-z0-9]+", " ", model_tail.lower()).strip()
        context_norm = re.sub(r"[^a-z0-9]+", " ", context_text).strip()
        tail_compact = re.sub(r"[^a-z0-9]+", "", model_tail.lower())
        context_compact = re.sub(r"[^a-z0-9]+", "", context_text)
        if (
            tail_norm
            and tail_norm not in context_norm
            and tail_compact
            and tail_compact not in context_compact
        ):
            retry_needed.append(
                "Evidence not model-specific enough; include explicit model name in title/snippet/url."
            )

    judge_status: str
    if hard_failures:
        judge_status = "rejected"
    elif retry_needed:
        judge_status = "needs_retry"
    else:
        judge_status = "accepted"

    # After final attempt unresolved rows are not silently dropped.
    if judge_status in {"rejected", "needs_retry"} and attempt_count >= max_attempts:
        judge_status = "unresolved"

    reasons = hard_failures + retry_needed
    judged["evidence_domain"] = evidence_domain
    judged["judge_status"] = judge_status
    judged["judge_reason"] = reasons[0] if reasons else "Accepted: date and evidence pass strict checks."
    judged["required_fix"] = " | ".join(reasons)
    judged["judge_confidence"] = (
        "0.95"
        if judge_status == "accepted"
        else ("0.90" if judge_status == "rejected" else ("0.60" if judge_status == "needs_retry" else "0.75"))
    )
    judged["judged_at_utc"] = now_utc_iso()

    return judged


def judge_rows(rows: list[dict[str, Any]], *, max_attempts: int) -> list[dict[str, Any]]:
    return [judge_row(row, max_attempts=max_attempts) for row in rows]


def build_review_rows(
    inventory_rows: list[dict[str, Any]],
    final_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    for item in inventory_rows:
        model_id = item["model_id"]
        row = final_by_model.get(model_id, {})
        review_rows.append(
            {
                "model_id": model_id,
                "org": item["org"],
                "launch_date": row.get("proposed_launch_date", "") if row.get("judge_status") == "accepted" else "",
                "evidence_url": row.get("evidence_url", ""),
                "evidence_title": row.get("evidence_title", ""),
                "evidence_published_date": row.get("evidence_published_date", ""),
                "judge_status": row.get("judge_status", "unresolved"),
                "judge_reason": row.get("judge_reason", "No judged row available."),
                "attempt_count": row.get("attempt_count", ""),
                "manual_override": "",
                "manual_notes": "",
            }
        )
    return review_rows


def build_candidates_rows(review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in review_rows
        if str(row.get("judge_status", "")).strip() in {"accepted", "unresolved"}
    ]


def build_canonical_rows(final_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    updated = now_utc_iso()
    for model_id in sorted(final_by_model.keys()):
        row = final_by_model[model_id]
        if row.get("judge_status") != "accepted":
            continue
        launch_date = str(row.get("proposed_launch_date", "")).strip()
        evidence_url = str(row.get("evidence_url", "")).strip()
        if not launch_date or not evidence_url:
            continue
        out.append(
            {
                "model_id": model_id,
                "org": row.get("org", ""),
                "launch_date": launch_date,
                "evidence_url": evidence_url,
                "evidence_title": row.get("evidence_title", ""),
                "evidence_published_date": row.get("evidence_published_date", ""),
                "evidence_type": row.get("evidence_type", ""),
                "judge_status": "accepted",
                "notes": row.get("notes", ""),
                "updated_at_utc": updated,
            }
        )
    return out


def command_inventory(args: argparse.Namespace) -> int:
    ensure_data_dir()
    inventory_rows, bucket_rows = scan_inventory(
        pathlib.Path(args.config),
        pathlib.Path(args.latest_aggregate),
        pathlib.Path(args.latest_responses),
        pathlib.Path(args.runs_dir),
    )

    write_csv(
        INVENTORY_CSV,
        [
            "model_id",
            "org",
            "present_in_latest",
            "present_in_config",
            "present_in_runs_history",
            "variant_count",
            "variants",
        ],
        inventory_rows,
    )
    write_csv(BUCKETS_CSV, ["model_id", "org", "collector_agent", "bucket"], bucket_rows)

    print(f"Inventory rows: {len(inventory_rows)}")
    print(f"- {INVENTORY_CSV}")
    print(f"- {BUCKETS_CSV}")
    return 0


def command_init_sources(args: argparse.Namespace) -> int:
    if not INVENTORY_CSV.exists() or not BUCKETS_CSV.exists():
        command_inventory(args)

    if SOURCES_CSV.exists() and not args.force:
        raise SystemExit(
            f"Sources file already exists: {SOURCES_CSV}. Use --force to overwrite."
        )

    inventory_rows = read_csv(INVENTORY_CSV)
    bucket_rows = read_csv(BUCKETS_CSV)
    rows = build_sources_template(inventory_rows, bucket_rows)
    write_csv(SOURCES_CSV, COLLECTOR_FIELDS, rows)
    print(f"Initialized source template: {SOURCES_CSV}")
    return 0


def command_collect(args: argparse.Namespace) -> int:
    if not INVENTORY_CSV.exists() or not BUCKETS_CSV.exists():
        command_inventory(args)

    inventory_rows = read_csv(INVENTORY_CSV)
    bucket_rows = read_csv(BUCKETS_CSV)

    inventory_by_model = {row["model_id"]: row for row in inventory_rows}
    bucket_by_model = {row["model_id"]: row for row in bucket_rows}
    source_rows = sources_by_model(pathlib.Path(args.sources))

    selected = sorted(inventory_by_model.keys())
    if args.agent:
        selected = [
            model_id
            for model_id in selected
            if bucket_by_model[model_id]["collector_agent"] == args.agent
        ]

    rows = collect_rows_for_models(
        selected,
        inventory_by_model,
        bucket_by_model,
        source_rows,
        attempt_count=max(1, int(args.attempt_count)),
    )

    output_path = pathlib.Path(args.output)
    write_csv(output_path, COLLECTOR_FIELDS + ["attempt_count"], rows)
    print(f"Collected rows: {len(rows)}")
    print(f"- {output_path}")
    return 0


def command_judge(args: argparse.Namespace) -> int:
    rows = read_csv(pathlib.Path(args.input))
    judged = judge_rows(rows, max_attempts=max(1, int(args.max_attempts)))

    output_path = pathlib.Path(args.output)
    write_csv(output_path, JUDGE_FIELDS, judged)
    print(f"Judged rows: {len(judged)}")
    print(f"- {output_path}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    ensure_data_dir()
    command_inventory(args)
    if not pathlib.Path(args.sources).exists():
        command_init_sources(argparse.Namespace(**{**vars(args), "force": False}))

    inventory_rows = read_csv(INVENTORY_CSV)
    bucket_rows = read_csv(BUCKETS_CSV)
    inventory_by_model = {row["model_id"]: row for row in inventory_rows}
    bucket_by_model = {row["model_id"]: row for row in bucket_rows}

    source_path = pathlib.Path(args.sources)
    source_rows = sources_by_model(source_path)

    pending = set(inventory_by_model.keys())
    final_by_model: dict[str, dict[str, Any]] = {}
    attempt_rows: list[dict[str, Any]] = []

    max_attempts = max(1, int(args.max_attempts))

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break

        selected = sorted(pending)
        collected_rows = collect_rows_for_models(
            selected,
            inventory_by_model,
            bucket_by_model,
            source_rows,
            attempt_count=attempt,
        )

        if attempt == max_attempts:
            write_csv(COLLECTION_CSV, COLLECTOR_FIELDS + ["attempt_count"], collected_rows)

        judged_rows = judge_rows(collected_rows, max_attempts=max_attempts)
        attempt_rows.extend(judged_rows)

        next_pending: set[str] = set()
        for row in judged_rows:
            model_id = row["model_id"]
            status = row["judge_status"]
            if status == "accepted":
                final_by_model[model_id] = row
                continue
            if status in {"needs_retry", "rejected"} and attempt < max_attempts:
                next_pending.add(model_id)
                # keep latest guidance for retry
                final_by_model[model_id] = row
            else:
                unresolved = dict(row)
                unresolved["judge_status"] = "unresolved"
                if not unresolved.get("judge_reason"):
                    unresolved["judge_reason"] = "Unresolved after max attempts."
                final_by_model[model_id] = unresolved

        pending = next_pending

    # Guarantee coverage for every inventory model.
    for model_id in inventory_by_model:
        if model_id in final_by_model:
            continue
        final_by_model[model_id] = {
            "model_id": model_id,
            "org": inventory_by_model[model_id]["org"],
            "proposed_launch_date": "",
            "evidence_url": "",
            "evidence_title": "",
            "evidence_published_date": "",
            "evidence_type": "",
            "notes": "",
            "judge_status": "unresolved",
            "judge_reason": "No attempt rows produced.",
            "attempt_count": 0,
        }

    final_rows = [final_by_model[model_id] for model_id in sorted(final_by_model.keys())]
    write_csv(JUDGED_CSV, JUDGE_FIELDS, final_rows)
    write_csv(ATTEMPTS_CSV, JUDGE_FIELDS, attempt_rows)

    review_rows = build_review_rows(inventory_rows, final_by_model)
    candidate_rows = build_candidates_rows(review_rows)
    canonical_rows = build_canonical_rows(final_by_model)

    write_csv(REVIEW_CSV, REVIEW_FIELDS, review_rows)
    write_csv(CANDIDATES_CSV, REVIEW_FIELDS, candidate_rows)
    write_csv(CANONICAL_CSV, CANONICAL_FIELDS, canonical_rows)

    accepted = sum(1 for row in final_rows if row.get("judge_status") == "accepted")
    unresolved = sum(1 for row in final_rows if row.get("judge_status") == "unresolved")
    print(f"Run complete: accepted={accepted} unresolved={unresolved} total={len(final_rows)}")
    print(f"- {COLLECTION_CSV}")
    print(f"- {JUDGED_CSV}")
    print(f"- {ATTEMPTS_CSV}")
    print(f"- {REVIEW_CSV}")
    print(f"- {CANDIDATES_CSV}")
    print(f"- {CANONICAL_CSV}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model launch-date provenance pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG))
    common.add_argument("--latest-aggregate", default=str(DEFAULT_LATEST_AGGREGATE))
    common.add_argument("--latest-responses", default=str(DEFAULT_LATEST_RESPONSES))
    common.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    subparsers.add_parser("inventory", parents=[common], help="Build tested model inventory and buckets")

    init_sources = subparsers.add_parser(
        "init-sources", parents=[common], help="Initialize source template CSV"
    )
    init_sources.add_argument("--force", action="store_true")

    collect = subparsers.add_parser("collect", parents=[common], help="Build collector rows")
    collect.add_argument("--sources", default=str(SOURCES_CSV))
    collect.add_argument("--output", default=str(COLLECTION_CSV))
    collect.add_argument("--agent", default="")
    collect.add_argument("--attempt-count", type=int, default=1)

    judge = subparsers.add_parser("judge", parents=[common], help="Judge collected rows")
    judge.add_argument("--input", default=str(COLLECTION_CSV))
    judge.add_argument("--output", default=str(JUDGED_CSV))
    judge.add_argument("--max-attempts", type=int, default=3)

    run = subparsers.add_parser("run", parents=[common], help="Run inventory+collect+judge loop")
    run.add_argument("--sources", default=str(SOURCES_CSV))
    run.add_argument("--max-attempts", type=int, default=3)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inventory":
        return command_inventory(args)
    if args.command == "init-sources":
        return command_init_sources(args)
    if args.command == "collect":
        return command_collect(args)
    if args.command == "judge":
        return command_judge(args)
    if args.command == "run":
        return command_run(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
