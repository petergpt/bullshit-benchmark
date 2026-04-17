#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, List


WIDTH = 1600
HEIGHT = 900
MARGIN_LEFT = 120
MARGIN_RIGHT = 180
MARGIN_TOP = 150
MARGIN_BOTTOM = 120

PLOT_LEFT = MARGIN_LEFT
PLOT_RIGHT = WIDTH - MARGIN_RIGHT
PLOT_TOP = MARGIN_TOP
PLOT_BOTTOM = HEIGHT - MARGIN_BOTTOM
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT
PLOT_HEIGHT = PLOT_BOTTOM - PLOT_TOP

BG = "#f6f1e7"
CARD = "#fffdf8"
GRID = "#ddd6c8"
GRID_SOFT = "#ebe4d7"
INK = "#14181a"
INK_SOFT = "#5b646a"
ACCENT = "#1e8f7b"
ACCENT_SOFT = "#d9f0eb"
MILESTONE = "#d6a21d"
MILESTONE_SOFT = "#fff1bf"
CURRENT = "#222222"


@dataclass
class StarEvent:
    timestamp: datetime
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a light-mode GitHub star milestone chart as SVG/PNG."
    )
    parser.add_argument(
        "--repo",
        default="petergpt/bullshit-benchmark",
        help="GitHub repo in owner/name form.",
    )
    parser.add_argument(
        "--milestone",
        type=int,
        default=1000,
        help="Milestone star count to highlight.",
    )
    parser.add_argument(
        "--input-json",
        default="",
        help="Optional path to pre-fetched stargazer JSON.",
    )
    parser.add_argument(
        "--output-svg",
        default="docs/images/bullshitbench-1000-stars-milestone.svg",
        help="Path to write the SVG chart.",
    )
    parser.add_argument(
        "--output-png",
        default="docs/images/bullshitbench-1000-stars-milestone.png",
        help="Path to write the PNG chart.",
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


def load_stargazers(repo: str, input_json: str) -> list[dict]:
    if input_json:
        with open(input_json, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return fetch_stargazers(repo)


def build_events(raw: Iterable[dict]) -> list[StarEvent]:
    ordered = sorted(raw, key=lambda item: item["starred_at"])
    events: list[StarEvent] = []
    for idx, item in enumerate(ordered, start=1):
        events.append(StarEvent(timestamp=iso_to_dt(item["starred_at"]), count=idx))
    return events


def nice_y_max(value: int) -> int:
    step = 100 if value <= 1500 else 250
    return int(math.ceil(value / step) * step)


def y_ticks(y_max: int) -> list[int]:
    if y_max <= 1200:
        return list(range(0, min(y_max, 1000) + 1, 200))
    step = 250 if y_max <= 2500 else 500
    return list(range(0, y_max + 1, step))


def format_day(dt: datetime) -> str:
    return f"{dt.strftime('%b')} {dt.day}"


def x_ticks(start: datetime, end: datetime) -> list[datetime]:
    first_day = datetime.combine(start.date(), time.min, tzinfo=timezone.utc)
    last_day = datetime.combine(end.date(), time.min, tzinfo=timezone.utc)
    total_days = max(1, (last_day.date() - first_day.date()).days)
    step_days = max(1, math.ceil(total_days / 6))
    ticks: list[datetime] = []
    cursor = first_day
    while cursor <= last_day:
        ticks.append(cursor)
        cursor += timedelta(days=step_days)
    if ticks[-1] != last_day:
        ticks.append(last_day)
    return ticks


def path_from_points(points: list[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)


def area_from_points(points: list[tuple[float, float]]) -> str:
    start_x = points[0][0]
    end_x = points[-1][0]
    commands = [f"M {start_x:.2f} {PLOT_BOTTOM:.2f}", f"L {start_x:.2f} {points[0][1]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    commands.append(f"L {end_x:.2f} {PLOT_BOTTOM:.2f}")
    commands.append("Z")
    return " ".join(commands)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_svg(repo: str, milestone_target: int, events: list[StarEvent]) -> str:
    if not events:
        raise ValueError("No stargazer events found")

    milestone_event = next((event for event in events if event.count >= milestone_target), None)
    if milestone_event is None:
        raise ValueError(f"Milestone {milestone_target} not reached; current count is {events[-1].count}")

    start = events[0].timestamp
    last = events[-1].timestamp
    span = max(last - start, timedelta(days=1))
    padded_end = last + max(span * 0.16, timedelta(days=2))
    y_max = nice_y_max(max(events[-1].count, milestone_target))

    def sx(value: datetime) -> float:
        fraction = (value - start).total_seconds() / (padded_end - start).total_seconds()
        return PLOT_LEFT + (fraction * PLOT_WIDTH)

    def sy(value: int) -> float:
        fraction = value / y_max
        return PLOT_BOTTOM - (fraction * PLOT_HEIGHT)

    line_points = [(sx(event.timestamp), sy(event.count)) for event in events]
    area_path = area_from_points(line_points)
    line_path = path_from_points(line_points)

    milestone_x = sx(milestone_event.timestamp)
    milestone_y = sy(milestone_event.count)
    current_x = sx(last)
    current_y = sy(events[-1].count)

    box_width = 320
    box_height = 100
    box_x = min(milestone_x + 34, PLOT_RIGHT - box_width + 30)
    box_y = max(PLOT_TOP + 10, milestone_y - 110)
    pointer_y = box_y + 62

    title = "BullshitBench crossed 1,000 GitHub stars"
    subtitle = (
        f"Milestone reached on {milestone_event.timestamp.strftime('%B')} "
        f"{milestone_event.timestamp.day}, {milestone_event.timestamp.year} "
        f"at {milestone_event.timestamp.strftime('%H:%M:%S')} UTC"
    )
    total_label = f"{events[-1].count:,} total stars"
    repo_label = repo
    current_label = (
        f"Current: {events[-1].count:,} as of "
        f"{last.strftime('%B')} {last.day}, {last.year}"
    )

    y_guides = []
    for tick in y_ticks(y_max):
        y = sy(tick)
        y_guides.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{PLOT_RIGHT}" y2="{y:.2f}" '
            f'stroke="{GRID_SOFT if tick else GRID}" stroke-width="1" />'
        )
        y_guides.append(
            f'<text x="{PLOT_LEFT - 18}" y="{y + 5:.2f}" text-anchor="end" '
            f'fill="{INK_SOFT}" font-size="20" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{tick:,}</text>'
        )

    tick_values = x_ticks(start, last)
    tick_positions = [(tick, sx(tick)) for tick in tick_values]
    if len(tick_positions) >= 2 and (tick_positions[-1][1] - tick_positions[-2][1]) < 105:
        tick_positions.pop(-2)

    x_guides = []
    for tick, x in tick_positions:
        x_guides.append(
            f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{PLOT_BOTTOM}" stroke="{GRID_SOFT}" stroke-width="1" />'
        )
        x_guides.append(
            f'<text x="{x:.2f}" y="{PLOT_BOTTOM + 38}" text-anchor="middle" '
            f'fill="{INK_SOFT}" font-size="20" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(format_day(tick))}</text>'
        )

    repo_pill_width = max(230, 90 + len(repo_label) * 8)
    total_pill_width = max(210, 110 + len(total_label) * 8)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{esc(title)}">
  <defs>
    <linearGradient id="bgGlow" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e8f4f0" />
      <stop offset="100%" stop-color="{BG}" />
    </linearGradient>
    <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{ACCENT}" stop-opacity="0.26" />
      <stop offset="100%" stop-color="{ACCENT}" stop-opacity="0.04" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="16" flood-color="#7f8d8c" flood-opacity="0.18" />
    </filter>
  </defs>

  <rect width="100%" height="100%" fill="url(#bgGlow)" />
  <circle cx="118" cy="96" r="140" fill="#efe6d2" opacity="0.9" />
  <circle cx="1480" cy="100" r="180" fill="#e3efe7" opacity="0.85" />

  <rect x="28" y="28" width="{WIDTH - 56}" height="{HEIGHT - 56}" rx="28" fill="{CARD}" stroke="{GRID}" filter="url(#shadow)" />

  <text x="{PLOT_LEFT}" y="88" fill="{INK}" font-size="48" font-weight="700" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(title)}</text>
  <text x="{PLOT_LEFT}" y="122" fill="{INK_SOFT}" font-size="24" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(subtitle)}</text>

  <rect x="{PLOT_LEFT}" y="36" width="{repo_pill_width}" height="34" rx="17" fill="{ACCENT_SOFT}" />
  <text x="{PLOT_LEFT + 18}" y="58" fill="{ACCENT}" font-size="18" font-weight="700" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(repo_label)}</text>

  <rect x="{WIDTH - total_pill_width - 52}" y="36" width="{total_pill_width}" height="44" rx="22" fill="#f4f0e4" stroke="{GRID}" />
  <text x="{WIDTH - total_pill_width / 2 - 52}" y="64" text-anchor="middle" fill="{INK}" font-size="24" font-weight="700" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(total_label)}</text>

  <text x="{PLOT_RIGHT}" y="{PLOT_BOTTOM + 84}" text-anchor="end" fill="{INK_SOFT}" font-size="18" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{esc(current_label)}</text>

  <rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" rx="24" fill="#fcfaf4" stroke="{GRID_SOFT}" />
  <text x="{PLOT_LEFT + 18}" y="{PLOT_TOP + 28}" fill="{INK_SOFT}" font-size="18" font-weight="600" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">GitHub stars</text>

  {''.join(y_guides)}
  {''.join(x_guides)}

  <line x1="{PLOT_LEFT}" y1="{milestone_y:.2f}" x2="{PLOT_RIGHT}" y2="{milestone_y:.2f}" stroke="{MILESTONE}" stroke-width="2" stroke-dasharray="8 8" />
  <line x1="{milestone_x:.2f}" y1="{PLOT_TOP}" x2="{milestone_x:.2f}" y2="{PLOT_BOTTOM}" stroke="{MILESTONE}" stroke-width="2" stroke-dasharray="8 8" opacity="0.7" />

  <path d="{area_path}" fill="url(#areaFill)" />
  <path d="{line_path}" fill="none" stroke="{ACCENT}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />

  <circle cx="{current_x:.2f}" cy="{current_y:.2f}" r="7" fill="{CURRENT}" />
  <circle cx="{milestone_x:.2f}" cy="{milestone_y:.2f}" r="12" fill="{MILESTONE_SOFT}" stroke="{MILESTONE}" stroke-width="4" />

  <path d="M {milestone_x + 14:.2f} {milestone_y:.2f} L {box_x:.2f} {pointer_y:.2f}" fill="none" stroke="{MILESTONE}" stroke-width="3" stroke-linecap="round" />
  <rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_width}" height="{box_height}" rx="18" fill="{MILESTONE_SOFT}" stroke="{MILESTONE}" stroke-width="2" />
  <text x="{box_x + 22:.2f}" y="{box_y + 36:.2f}" fill="{INK}" font-size="28" font-weight="700" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">1,000 stars</text>
  <text x="{box_x + 22:.2f}" y="{box_y + 64:.2f}" fill="{INK_SOFT}" font-size="18" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">Reached {esc(format_day(milestone_event.timestamp))}, {milestone_event.timestamp.year}</text>
  <text x="{box_x + 22:.2f}" y="{box_y + 86:.2f}" fill="{INK_SOFT}" font-size="18" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">{milestone_event.timestamp.strftime('%H:%M:%S')} UTC</text>

  <text x="{PLOT_LEFT}" y="{HEIGHT - 36}" fill="{INK_SOFT}" font-size="16" font-family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif">Source: GitHub stargazer events API · Rendered as a one-off share image</text>
</svg>
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_png(svg_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsvg-convert",
            str(svg_path),
            "-o",
            str(png_path),
            "-w",
            str(WIDTH),
            "-h",
            str(HEIGHT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    args = parse_args()
    raw = load_stargazers(args.repo, args.input_json)
    events = build_events(raw)
    svg = render_svg(args.repo, args.milestone, events)

    svg_path = Path(args.output_svg)
    png_path = Path(args.output_png)
    write_text(svg_path, svg)
    render_png(svg_path, png_path)

    milestone_event = next(event for event in events if event.count >= args.milestone)
    print(f"repo={args.repo}")
    print(f"current_stars={events[-1].count}")
    print(f"milestone={args.milestone}")
    print(f"milestone_timestamp_utc={milestone_event.timestamp.isoformat().replace('+00:00', 'Z')}")
    print(f"svg={svg_path.resolve()}")
    print(f"png={png_path.resolve()}")


if __name__ == "__main__":
    main()
