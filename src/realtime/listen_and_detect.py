from __future__ import annotations

import argparse
import json
import queue
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sounddevice as sd

from src.utils.audio_features import extract_features


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


def log_event(db_path: Path, location: str, prob: float) -> None:
	ts = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(db_path) as conn:
		conn.execute("INSERT INTO events (timestamp_utc, location, probability) VALUES (?, ?, ?)", (ts, location, float(prob)))
		conn.commit()


def load_model(models_dir: Path):
	model_path = models_dir / "laughter_rf.joblib"
	if not model_path.exists():
		raise FileNotFoundError("Model not found. Train with: python -m src.models.train")
	return joblib.load(model_path)


def stream_and_detect(args) -> None:
	root = Path(__file__).resolve().parents[2]
	db_path = root / "data" / "events.sqlite"
	models_dir = root / "models"
	init_db(db_path)

	clf = load_model(models_dir)

	# Audio params
	sr = 16000
	window_seconds = args.window
	hop_seconds = args.hop
	block_size = int(hop_seconds * sr)

	audio_buffer = np.zeros(int(window_seconds * sr), dtype=np.float32)
	q_in: queue.Queue[np.ndarray] = queue.Queue()

	def audio_callback(indata, frames, time_info, status):  # type: ignore[no-redef]
		if status:
			print(status, file=sys.stderr)
		q_in.put(indata.copy().ravel())

	print("Listening... Press Ctrl+C to stop.")
	with sd.InputStream(channels=1, samplerate=sr, blocksize=block_size, callback=audio_callback):
		while True:
			try:
				block = q_in.get(timeout=1.0)
			except queue.Empty:
				continue

			# Update rolling buffer
			audio_buffer = np.roll(audio_buffer, -len(block))
			audio_buffer[-len(block):] = block

			# Classify current window
			feat = extract_features(audio_buffer.astype(np.float64), sr)
			prob = float(clf.predict_proba([feat])[0][1])
			if prob >= args.threshold:
				log_event(db_path, args.location, prob)
				print(f"Laughter detected at {datetime.now().isoformat()} p={prob:.2f} @ {args.location}")

			time.sleep(0.01)


def get_known_locations() -> dict[str, dict[str, float]]:
	root = Path(__file__).resolve().parents[2]
	cfg = root / "config" / "locations.json"
	if cfg.exists():
		return json.loads(cfg.read_text(encoding="utf-8"))
	return {}


def parse_args():
	parser = argparse.ArgumentParser(description="Real-time laughter detection and logging")
	parser.add_argument("--location", type=str, required=True, help="Location label to attach to events")
	parser.add_argument("--threshold", type=float, default=0.8, help="Probability threshold for event logging")
	parser.add_argument("--window", type=float, default=2.0, help="Seconds per classification window")
	parser.add_argument("--hop", type=float, default=1.0, help="Hop seconds between windows")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	known = get_known_locations()
	if args.location not in known:
		print(f"Warning: location '{args.location}' not in config/locations.json. It will still log but won't map.")
	stream_and_detect(args)

