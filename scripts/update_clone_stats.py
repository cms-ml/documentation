#!/usr/bin/env python3
"""Record clone traffic for the CMS skills marketplace repository.

GitHub keeps only the last 14 days of clone traffic, so each reading is merged
into a running record that the marketplace page plots.

Usage:
    gh api repos/cms-ml/marketplace/traffic/clones > clones-raw.json
    python3 scripts/update_clone_stats.py clones-raw.json

Pass "-" instead of a file name to read the API response from stdin. Reading
the traffic API needs write access to cms-ml/marketplace.

The record is rewritten only when the data changes, so the scheduled workflow
commits only new readings. Exit code 0 on success, 1 on a bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "content" / "marketplace" / "data" / "clones.json"
REPOSITORY = "cms-ml/marketplace"
SOURCE = "https://docs.github.com/rest/metrics/traffic#get-repository-clones"


def load_response(source: str) -> dict:
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    try:
        response = json.loads(text)
    except json.JSONDecodeError:
        response = None
    if not isinstance(response, dict) or "clones" not in response:
        print(f"error: not a traffic/clones API response: {text[:200]!r}", file=sys.stderr)
        sys.exit(1)
    return response


def merge(record: dict, response: dict) -> dict:
    """Return a new record with the response's days added or updated."""
    days = {d["date"]: d for d in record.get("days", [])}
    window = record.get("window")

    readings = sorted(response["clones"], key=lambda c: c["timestamp"])
    if readings:
        start = date.fromisoformat(readings[0]["timestamp"][:10])
        end = date.fromisoformat(readings[-1]["timestamp"][:10])
        # GitHub can leave out days without clones; inside the window they are zero
        day = start
        while day <= end:
            key = day.isoformat()
            days[key] = {"date": key, "count": 0, "uniques": 0}
            day += timedelta(days=1)
        # Newer readings win: they include the end of a day that was still running
        for c in readings:
            key = c["timestamp"][:10]
            days[key] = {"date": key, "count": c["count"], "uniques": c["uniques"]}
        window = {
            "days": 14,
            "end": end.isoformat(),
            "count": response["count"],
            "uniques": response["uniques"],
        }

    return {
        "repository": REPOSITORY,
        "source": SOURCE,
        "updated": record.get("updated"),
        "window": window,
        "days": [days[k] for k in sorted(days)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("response", help='traffic/clones API response file, or "-" for stdin')
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="record to update")
    args = parser.parse_args()

    old = json.loads(args.record.read_text()) if args.record.exists() else {}
    new = merge(old, load_response(args.response))

    if new["days"] == old.get("days") and new["window"] == old.get("window"):
        print(f"{args.record.name}: no change ({len(new['days'])} days)")
        return 0

    new["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(new, indent=2) + "\n")

    old_dates = {d["date"] for d in old.get("days", [])}
    added = sum(d["date"] not in old_dates for d in new["days"])
    first, last = new["days"][0]["date"], new["days"][-1]["date"]
    print(f"{args.record.name}: {len(new['days'])} days ({first} to {last}), {added} new")
    return 0


if __name__ == "__main__":
    sys.exit(main())
