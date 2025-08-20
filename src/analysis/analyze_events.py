from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_events(db_path: Path) -> pd.DataFrame:
	with sqlite3.connect(db_path) as conn:
		df = pd.read_sql_query("SELECT * FROM events", conn, parse_dates=["timestamp_utc"])  # type: ignore[arg-type]
	return df


def ensure_outputs(out_dir: Path) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)


def plot_by_hour(df: pd.DataFrame, out_dir: Path) -> None:
	df = df.copy()
	df["hour"] = df["timestamp_utc"].dt.hour
	plt.figure(figsize=(8, 4))
	sns.countplot(x="hour", data=df, color="#3b82f6")
	plt.title("Laughter events by hour of day")
	plt.xlabel("Hour")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(out_dir / "laughter_by_hour.png", dpi=150)
	plt.close()


def plot_daily_series(df: pd.DataFrame, out_dir: Path) -> None:
	daily = df.set_index("timestamp_utc").resample("D")["id"].count()
	plt.figure(figsize=(9, 4))
	daily.plot()
	plt.title("Daily laughter event counts")
	plt.xlabel("Date")
	plt.ylabel("Events")
	plt.tight_layout()
	plt.savefig(out_dir / "laughter_timeseries_daily.png", dpi=150)
	plt.close()


def build_map(df: pd.DataFrame, locations_cfg: dict, out_dir: Path) -> None:
	loc_counts = df.groupby("location")["id"].count().sort_values(ascending=False)
	if loc_counts.empty:
		print("No events to map.")
		return

	# Center map at weighted average of coordinates
	cent_lat, cent_lon, total = 0.0, 0.0, 0
	for loc, count in loc_counts.items():
		coord = locations_cfg.get(loc)
		if not coord:
			continue
		cent_lat += coord["lat"] * count
		cent_lon += coord["lon"] * count
		total += count
	if total == 0:
		cent_lat, cent_lon = 0.0, 0.0

	m = folium.Map(location=[cent_lat / max(1, total), cent_lon / max(1, total)], zoom_start=12 if total > 0 else 2)

	for loc, count in loc_counts.items():
		coord = locations_cfg.get(loc)
		if not coord:
			continue
		size = 5 + 3 * count
		folium.CircleMarker(
			location=[coord["lat"], coord["lon"]],
			radius=size,
			popup=f"{loc}: {count}",
			color="#ef4444",
			fill=True,
			fill_opacity=0.6,
		).add_to(m)

	m.save(out_dir / "laughter_map.html")


def main() -> None:
	root = Path(__file__).resolve().parents[2]
	db_path = root / "data" / "events.sqlite"
	out_dir = root / "outputs"
	locations_path = root / "config" / "locations.json"

	if not db_path.exists():
		print("No events database found. Run the real-time detector first.")
		return

	ensure_outputs(out_dir)
	df = load_events(db_path)
	if df.empty:
		print("No events logged yet.")
		return

	plot_by_hour(df, out_dir)
	plot_daily_series(df, out_dir)

	locations_cfg = json.loads(locations_path.read_text(encoding="utf-8")) if locations_path.exists() else {}
	build_map(df, locations_cfg, out_dir)
	print("Wrote plots and map to:", out_dir)


if __name__ == "__main__":
	main()

