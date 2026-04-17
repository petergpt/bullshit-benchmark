#!/usr/bin/env python3
"""Transform BullshitBench aggregate rows into Forge pairwise feedback events."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_API_BASE = "https://forge-api.arena.ai"
DEFAULT_DATASET = "data/v2/latest/aggregate.jsonl"
DEFAULT_CATEGORY = "bullshit_detection"
DEFAULT_VISIBILITY = "public"


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def infer_benchmark_version(dataset_path: pathlib.Path) -> str:
    normalized = str(dataset_path).replace("\\", "/").lower()
    if "/data/v2/" in normalized or "v2" in dataset_path.name.lower():
        return "v2"
    if "/data/latest/" in normalized or "v1" in dataset_path.name.lower():
        return "v1"
    return slugify(dataset_path.parent.name or dataset_path.stem)


def infer_arena_name(version: str) -> str:
    if version.startswith("v"):
        return f"BullshitBench {version.upper()}"
    return f"BullshitBench {version}"


@dataclass(frozen=True)
class AggregateRow:
    sample_id: str
    model: str
    question_id: str
    question: str
    domain: str
    technique: str
    consensus_score: float
    row_errors: tuple[str, ...]
    row_identity_mismatch: bool


class ForgeError(RuntimeError):
    pass


class ForgeClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | list[Any] | None = None,
        require_auth: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "bullshitbench-forge-uploader/1.0",
        }
        if require_auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8", "replace").strip()
                if not raw:
                    return None
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ForgeError(f"{method} {path} failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ForgeError(f"{method} {path} failed: {exc.reason}") from exc


def load_aggregate_rows(dataset_path: pathlib.Path) -> dict[str, list[AggregateRow]]:
    grouped: dict[str, dict[str, AggregateRow]] = defaultdict(dict)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            obj = json.loads(line)
            if obj.get("consensus_score") is None:
                continue

            row = AggregateRow(
                sample_id=str(obj["sample_id"]),
                model=str(obj["model"]),
                question_id=str(obj["question_id"]),
                question=str(obj["question"]),
                domain=str(obj.get("domain", "")),
                technique=str(obj.get("technique", "")),
                consensus_score=float(obj["consensus_score"]),
                row_errors=tuple(str(item) for item in obj.get("row_errors", []) or []),
                row_identity_mismatch=bool(obj.get("row_identity_mismatch")),
            )

            # Keep the last row for a given question/model if a dataset contains duplicates.
            grouped[row.question_id][row.model] = row

    return {
        question_id: sorted(models.values(), key=lambda row: row.model)
        for question_id, models in grouped.items()
    }


def build_event(
    row_a: AggregateRow,
    row_b: AggregateRow,
    *,
    category: str,
    benchmark_version: str,
    extra_tags: tuple[str, ...],
) -> dict[str, Any]:
    score_a = row_a.consensus_score
    score_b = row_b.consensus_score
    if score_a > score_b:
        winner = "system_a"
    elif score_b > score_a:
        winner = "system_b"
    else:
        winner = "tie"

    tags = [
        f"benchmark_{benchmark_version}",
        f"question_{row_a.question_id}",
        f"technique_{slugify(row_a.technique)}",
        f"domain_{slugify(row_a.domain)}",
    ]
    for tag in extra_tags:
        tag_text = str(tag).strip()
        if tag_text and tag_text not in tags:
            tags.append(tag_text)

    return {
        "type": "pairwise",
        "system_a": row_a.model,
        "system_b": row_b.model,
        "winner": winner,
        "category": category,
        "prompt": row_a.question,
        "tags": tags,
        "signals": {
            "model_a_consensus_score": score_a,
            "model_b_consensus_score": score_b,
            "score_margin": round(abs(score_a - score_b), 4),
        },
        "metadata": {
            "benchmark": "BullshitBench",
            "benchmark_version": benchmark_version,
            "question_id": row_a.question_id,
            "domain": row_a.domain,
            "technique": row_a.technique,
            "model_a_sample_id": row_a.sample_id,
            "model_b_sample_id": row_b.sample_id,
            "row_a_has_errors": bool(row_a.row_errors or row_a.row_identity_mismatch),
            "row_b_has_errors": bool(row_b.row_errors or row_b.row_identity_mismatch),
        },
    }


def iter_feedback_events(
    grouped_rows: dict[str, list[AggregateRow]],
    *,
    category: str,
    benchmark_version: str,
    extra_tags: tuple[str, ...],
    max_events: int | None = None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    for question_id in sorted(grouped_rows):
        rows = grouped_rows[question_id]
        for row_a, row_b in itertools.combinations(rows, 2):
            yield build_event(
                row_a,
                row_b,
                category=category,
                benchmark_version=benchmark_version,
                extra_tags=extra_tags,
            )
            emitted += 1
            if max_events is not None and emitted >= max_events:
                return


def summarize_events(grouped_rows: dict[str, list[AggregateRow]], max_events: int | None = None) -> dict[str, Any]:
    total_rows = sum(len(rows) for rows in grouped_rows.values())
    model_names = {row.model for rows in grouped_rows.values() for row in rows}
    tie_events = 0
    total_events = 0
    question_pairs = {}
    for question_id, rows in grouped_rows.items():
        pair_count = len(rows) * (len(rows) - 1) // 2
        question_pairs[question_id] = pair_count
    for question_id in sorted(grouped_rows):
        rows = grouped_rows[question_id]
        for row_a, row_b in itertools.combinations(rows, 2):
            total_events += 1
            if row_a.consensus_score == row_b.consensus_score:
                tie_events += 1
            if max_events is not None and total_events >= max_events:
                return {
                    "models": len(model_names),
                    "questions": len(grouped_rows),
                    "rows": total_rows,
                    "events": total_events,
                    "tie_events": tie_events,
                    "truncated": True,
                }
    return {
        "models": len(model_names),
        "questions": len(grouped_rows),
        "rows": total_rows,
        "events": total_events,
        "tie_events": tie_events,
        "truncated": False,
    }


def write_events(path: pathlib.Path, events: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def extract_list(payload: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    raise ForgeError(f"Could not extract list from payload: {payload!r}")


def extract_object(payload: Any, preferred_keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(payload, dict) and "id" in payload:
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, dict) and "id" in value:
                return value
    raise ForgeError(f"Could not extract object from payload: {payload!r}")


def ensure_workspace(client: ForgeClient, workspace_id: str | None, workspace_name: str) -> dict[str, Any]:
    if workspace_id:
        payload = client.request("GET", f"/v1/workspaces/{workspace_id}")
        return extract_object(payload, ("workspace", "data"))

    workspaces_payload = client.request("GET", "/v1/workspaces")
    workspaces = extract_list(workspaces_payload, ("workspaces", "items", "data"))
    for workspace in workspaces:
        if str(workspace.get("name", "")).strip().lower() == workspace_name.strip().lower():
            return workspace

    payload = {
        "name": workspace_name,
        "description": "Workspace for BullshitBench-derived Forge arenas.",
    }
    created = client.request("POST", "/v1/workspaces", payload)
    return extract_object(created, ("workspace", "data"))


def ensure_arena(
    client: ForgeClient,
    *,
    workspace_id: str,
    arena_id: str | None,
    arena_name: str,
    category: str,
    visibility: str,
    benchmark_version: str,
) -> dict[str, Any]:
    if arena_id:
        payload = client.request("GET", f"/v1/workspaces/{workspace_id}/arenas/{arena_id}")
        return extract_object(payload, ("arena", "data"))

    arenas_payload = client.request("GET", f"/v1/workspaces/{workspace_id}/arenas")
    arenas = extract_list(arenas_payload, ("arenas", "items", "data"))
    for arena in arenas:
        if str(arena.get("name", "")).strip().lower() == arena_name.strip().lower():
            return arena

    payload = {
        "name": arena_name,
        "description": (
            f"Pairwise leaderboard derived from BullshitBench {benchmark_version.upper()} "
            "published aggregate results."
        ),
        "categories": [category],
        "default_category": category,
        "visibility": visibility,
    }
    created = client.request("POST", f"/v1/workspaces/{workspace_id}/arenas", payload)
    return extract_object(created, ("arena", "data"))


def upload_events(
    client: ForgeClient,
    *,
    arena_id: str,
    events: Iterable[dict[str, Any]],
    progress_every: int,
) -> int:
    uploaded = 0
    for event in events:
        client.request("POST", f"/v1/arenas/{arena_id}/feedback", event)
        uploaded += 1
        if progress_every > 0 and uploaded % progress_every == 0:
            print(f"uploaded {uploaded} feedback events", file=sys.stderr)
    return uploaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push BullshitBench aggregate data into a Forge arena as pairwise feedback."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to BullshitBench aggregate.jsonl")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Forge API base URL")
    parser.add_argument(
        "--api-key-env",
        default="FORGE_API_KEY",
        help="Environment variable holding the Forge API key",
    )
    parser.add_argument("--workspace-id", help="Use an existing Forge workspace id")
    parser.add_argument("--workspace-name", default="BullshitBench", help="Workspace name to find or create")
    parser.add_argument("--arena-id", help="Use an existing Forge arena id")
    parser.add_argument("--arena-name", help="Arena name to find or create")
    parser.add_argument(
        "--direct-arena",
        action="store_true",
        help="Push directly to --arena-id without workspace or arena discovery",
    )
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Forge feedback category")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Append an extra tag to every emitted feedback event",
    )
    parser.add_argument(
        "--visibility",
        default=DEFAULT_VISIBILITY,
        choices=("public", "private"),
        help="Visibility for a newly created arena",
    )
    parser.add_argument("--max-events", type=int, help="Limit the number of feedback events to emit/upload")
    parser.add_argument("--progress-every", type=int, default=250, help="Progress interval during upload")
    parser.add_argument("--write-events", help="Write derived Forge feedback events to JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Transform and summarize without uploading")
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Upload feedback but skip leaderboard recompute/publish calls",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = pathlib.Path(args.dataset).resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    benchmark_version = infer_benchmark_version(dataset_path)
    arena_name = args.arena_name or infer_arena_name(benchmark_version)
    extra_tags = tuple(str(tag).strip() for tag in args.tag if str(tag).strip())

    grouped_rows = load_aggregate_rows(dataset_path)
    summary = summarize_events(grouped_rows, max_events=args.max_events)

    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "benchmark_version": benchmark_version,
                "workspace_name": args.workspace_name,
                "arena_name": arena_name,
                "category": args.category,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.write_events:
        output_path = pathlib.Path(args.write_events).resolve()
        written = write_events(
            output_path,
            iter_feedback_events(
                grouped_rows,
                category=args.category,
                benchmark_version=benchmark_version,
                extra_tags=extra_tags,
                max_events=args.max_events,
            ),
        )
        print(f"wrote {written} events to {output_path}")

    if args.dry_run:
        return 0

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing Forge API key in environment variable {args.api_key_env}")

    client = ForgeClient(args.api_base, api_key)
    if args.direct_arena:
        if not args.arena_id:
            raise SystemExit("--direct-arena requires --arena-id")
        arena_id = str(args.arena_id)
        print(f"using direct arena {arena_id}")
    else:
        workspace = ensure_workspace(client, args.workspace_id, args.workspace_name)
        workspace_id = str(workspace["id"])
        arena = ensure_arena(
            client,
            workspace_id=workspace_id,
            arena_id=args.arena_id,
            arena_name=arena_name,
            category=args.category,
            visibility=args.visibility,
            benchmark_version=benchmark_version,
        )
        arena_id = str(arena["id"])

        print(f"using workspace {workspace_id} ({workspace.get('name', args.workspace_name)})")
        print(f"using arena {arena_id} ({arena.get('name', arena_name)})")

    uploaded = upload_events(
        client,
        arena_id=arena_id,
        events=iter_feedback_events(
            grouped_rows,
            category=args.category,
            benchmark_version=benchmark_version,
            extra_tags=extra_tags,
            max_events=args.max_events,
        ),
        progress_every=args.progress_every,
    )
    print(f"uploaded {uploaded} feedback events")

    if not args.skip_publish:
        client.request("POST", f"/v1/arenas/{arena_id}/leaderboard/recompute")
        client.request("POST", f"/v1/arenas/{arena_id}/leaderboard/publish", payload={})
        leaderboard = client.request("GET", f"/v1/arenas/{arena_id}/leaderboard", require_auth=False)
        print(
            json.dumps(
                {
                    "arena_id": arena_id,
                    "leaderboard_preview": leaderboard,
                },
                indent=2,
                sort_keys=True,
            )
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ForgeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
