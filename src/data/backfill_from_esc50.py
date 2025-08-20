from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def init_db(db_path: Path) -> None:
	db_path.parent.mkdir(parents=True, exist_ok=True)
	with sqlite3.connect(db_path) as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp_utc TEXT NOT NULL,
				location TEXT NOT NULL,
				probability REAL NOT NULL
			);
			"""
		)
		conn.commit()


def log_event(db_path: Path, when: datetime, location: str, prob: float) -> None:
	with sqlite3.connect(db_path) as conn:
		conn.execute("INSERT INTO events (timestamp_utc, location, probability) VALUES (?, ?, ?)", (when.isoformat(), location, float(prob)))
		conn.commit()


def load_esc50_metadata(esc_root: Path) -> list[dict]:
	meta_path = esc_root / "meta" / "esc50.csv"
	rows: list[dict] = []
	with open(meta_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Backfill laughter events from ESC-50 for demo/analysis")
	parser.add_argument("--location", type=str, default="Park", help="Location label for backfilled events")
	parser.add_argument("--days", type=int, default=7, help="Distribute events over the past N days")
	parser.add_argument("--max_events", type=int, default=250, help="Maximum events to insert")
	args = parser.parse_args()

	root = Path(__file__).resolve().parents[2]
	db_path = root / "data" / "events.sqlite"
	esc_root = root / "data" / "raw" / "ESC-50-master"
	if not esc_root.exists():
		raise FileNotFoundError("ESC-50 not found. Run: python -m src.data.download_esc50")

	init_db(db_path)
	meta = load_esc50_metadata(esc_root)
	laugh_rows = [r for r in meta if r.get("category") == "laughing"]
	if not laugh_rows:
		raise RuntimeError("No 'laughing' rows found in ESC-50 metadata")

	random.seed(42)
	now = datetime.now(timezone.utc)
	start = now - timedelta(days=args.days)

	to_insert = min(len(laugh_rows), args.max_events)
	for _ in range(to_insert):
		delta = random.random() * (now - start).total_seconds()
		when = start + timedelta(seconds=delta)
		prob = 0.85 + 0.15 * random.random()
		log_event(db_path, when, args.location, prob)

	print(f"Inserted {to_insert} demo events at location '{args.location}' into {db_path}")


if __name__ == "__main__":
	main()

