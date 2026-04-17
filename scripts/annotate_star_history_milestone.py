#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = "https://api.star-history.com/svg?repos={repo}&type=Date"
STROKE = "#dd4528"

# Star History default light card geometry for the Date chart currently returned
# by api.star-history.com for a single repo.
ROOT_WIDTH = 800
ROOT_HEIGHT = 533.333
OUTER_GROUP_TX = 70
OUTER_GROUP_TY = 60
PLOT_WIDTH = 700
PLOT_HEIGHT = 423.833
Y_MAX = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a one-off milestone marker onto a Star History SVG."
    )
    parser.add_argument("--repo", default="petergpt/bullshit-benchmark")
    parser.add_argument("--milestone", type=int, default=1000)
    parser.add_argument(
        "--output-svg",
        default="docs/images/bullshitbench-star-history-1000-stars.svg",
    )
    parser.add_argument(
        "--output-png",
        default="docs/images/bullshitbench-star-history-1000-stars.png",
    )
    return parser.parse_args()


def iso_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_stargazers(repo: str) -> list[dict]:
    command = [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github.star+json",
        f"repos/{repo}/stargazers",
        "--paginate",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def fetch_star_history_svg(repo: str) -> str:
    result = subprocess.run(
        ["curl", "-sL", SOURCE_URL.format(repo=repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def milestone_event(raw: list[dict], milestone: int) -> tuple[datetime, int]:
    ordered = sorted(raw, key=lambda item: item["starred_at"])
    if len(ordered) < milestone:
        raise SystemExit(f"Milestone {milestone} not reached. Current stars: {len(ordered)}")
    return iso_to_dt(ordered[milestone - 1]["starred_at"]), len(ordered)


def root_coords(start: datetime, end: datetime, moment: datetime, count: int) -> tuple[float, float]:
    span = (end - start).total_seconds() or 1
    x_ratio = (moment - start).total_seconds() / span
    x = OUTER_GROUP_TX + (x_ratio * PLOT_WIDTH)
    y = OUTER_GROUP_TY + (PLOT_HEIGHT - ((count / Y_MAX) * PLOT_HEIGHT))
    return x, y


def make_overlay(marker_x: float, marker_y: float, milestone: int, date_label: str) -> str:
    dot_x = marker_x - 2
    dot_y = marker_y + 1.2
    box_w = 178
    box_h = 34
    box_x = 540
    box_y = 86
    label_x = box_x + 29
    label_y = box_y + 22
    square_x = box_x + 14
    square_y = box_y + 14
    elbow_x = 705
    elbow_y = 96
    line = (
        f'<path d="M {box_x + box_w:.1f} {box_y + box_h / 2:.1f} '
        f'L {elbow_x:.1f} {elbow_y:.1f} L {dot_x:.1f} {dot_y:.1f}" '
        f'fill="none" stroke="#000" stroke-width="2" filter="url(#xkcdify)"/>'
    )
    rect = (
        f'<rect width="{box_w}" height="{box_h}" x="{box_x}" y="{box_y}" '
        f'fill-opacity=".88" stroke="#000" stroke-width="2" filter="url(#xkcdify)" '
        f'rx="5" ry="5" style="fill:#fff"/>'
    )
    label_square = (
        f'<rect width="8" height="8" x="{square_x}" y="{square_y}" '
        f'filter="url(#xkcdify)" rx="2" ry="2" style="fill:{STROKE}"/>'
    )
    label = (
        f'<text x="{label_x}" y="{label_y}" '
        f'style="font-family:xkcd;font-size:15px;fill:#000">{milestone:,} stars</text>'
    )
    sublabel = (
        f'<text x="{box_x + 8}" y="{box_y + 48}" '
        f'style="font-family:xkcd;font-size:12px;fill:#666">{date_label}</text>'
    )
    dot = (
        f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="6.5" fill="#fff" '
        f'stroke="{STROKE}" stroke-width="3"/>'
    )
    halo = (
        f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="10.5" fill="none" '
        f'stroke="{STROKE}" stroke-width="1.5" opacity="0.45" stroke-dasharray="3 4"/>'
    )
    return f'<g id="milestone-overlay">{line}{rect}{label_square}{label}{dot}{halo}</g>'


def patch_svg(svg: str, overlay: str) -> str:
    if "</filter><g pointer-events=\"all\"" not in svg:
        raise SystemExit("Unexpected Star History SVG structure; filter anchor not found.")
    svg = svg.replace(
        "</filter><g pointer-events=\"all\"",
        f'</filter><rect width="{ROOT_WIDTH}" height="{ROOT_HEIGHT}" fill="#fff"/><g pointer-events="all"',
        1,
    )
    if not svg.endswith("</svg>"):
        raise SystemExit("Unexpected SVG ending.")
    return svg[:-6] + overlay + "</svg>"


def rasterize(svg_path: Path, png_path: Path) -> None:
    subprocess.run(
        [
            "rsvg-convert",
            str(svg_path),
            "-o",
            str(png_path),
            "-w",
            str(ROOT_WIDTH),
            "-h",
            "533",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    args = parse_args()
    raw = fetch_stargazers(args.repo)
    milestone_dt, current_stars = milestone_event(raw, args.milestone)
    ordered = sorted(raw, key=lambda item: item["starred_at"])
    start = iso_to_dt(ordered[0]["starred_at"])
    end = iso_to_dt(ordered[-1]["starred_at"])
    marker_x, marker_y = root_coords(start, end, milestone_dt, args.milestone)
    date_label = f"{milestone_dt.strftime('%b')} {milestone_dt.day}"

    base_svg = fetch_star_history_svg(args.repo)
    overlay = make_overlay(marker_x, marker_y, args.milestone, date_label)
    final_svg = patch_svg(base_svg, overlay)

    svg_path = Path(args.output_svg)
    png_path = Path(args.output_png)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(final_svg, encoding="utf-8")
    rasterize(svg_path, png_path)

    print(f"repo={args.repo}")
    print(f"current_stars={current_stars}")
    print(f"milestone_timestamp_utc={milestone_dt.isoformat().replace('+00:00', 'Z')}")
    print(f"svg={svg_path.resolve()}")
    print(f"png={png_path.resolve()}")


if __name__ == "__main__":
    main()
