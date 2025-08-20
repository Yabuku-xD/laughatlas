from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.audio_features import extract_features


def load_esc50_metadata(esc_root: Path) -> list[dict]:
	meta_path = esc_root / "meta" / "esc50.csv"
	rows: list[dict] = []
	with open(meta_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)
	return rows



TARGET_SR = 16000


def load_audio_file(path: Path) -> Tuple[np.ndarray, int]:
	audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
	return audio, sr


def main() -> None:
	root = Path(__file__).resolve().parents[2]
	raw_dir = root / "data" / "raw" / "ESC-50-master" / "audio"
	meta_root = root / "data" / "raw" / "ESC-50-master"
	models_dir = root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	if not meta_root.exists():
		raise FileNotFoundError("ESC-50 not found. Run: python -m src.data.download_esc50")

	meta = load_esc50_metadata(meta_root)

	X: list[np.ndarray] = []
	y: list[int] = []

	print("Extracting features (this can take a few minutes)...")
	for row in tqdm(meta):
		label = row["category"]
		filepath = raw_dir / row["filename"]
		try:
			audio, sr = load_audio_file(filepath)
			feat = extract_features(audio, sr)
			X.append(feat)
			y.append(1 if label == "laughing" else 0)
		except Exception as e:
			print("Failed on", filepath, e)

	X_arr = np.vstack(X)
	y_arr = np.array(y)

	X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr)

	clf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced")
	clf.fit(X_train, y_train)

	print("Eval on holdout:")
	pred = clf.predict(X_test)
	print(classification_report(y_test, pred, digits=3))

	model_path = models_dir / "laughter_rf.joblib"
	feat_info_path = models_dir / "feature_dim.txt"
	joblib.dump(clf, model_path)
	feat_info_path.write_text(str(X_arr.shape[1]))
	print("Saved:", model_path)


if __name__ == "__main__":
	main()

