import numpy as np
import librosa


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
	"""Compute MFCCs and spectral features from mono audio.

	Returns a 1D feature vector aggregated over time.
	"""
	if audio.ndim > 1:
		audio = np.mean(audio, axis=0)
	# Ensure finite
	audio = np.nan_to_num(audio)

	# MFCCs
	mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
	mfcc_delta = librosa.feature.delta(mfcc)
	mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

	# Spectral features
	spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
	spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
	spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y=audio)

	# Aggregate over time (mean and std)
	def agg(feat: np.ndarray) -> np.ndarray:
		return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])

	features = [
		agg(mfcc),
		agg(mfcc_delta),
		agg(mfcc_delta2),
		agg(spec_centroid),
		agg(spec_bw),
		agg(spec_contrast),
		agg(spec_rolloff),
		agg(zcr),
	]

	return np.concatenate(features)


def frame_audio(audio: np.ndarray, sr: int, window_seconds: float, hop_seconds: float) -> list[tuple[np.ndarray, int]]:
	"""Yield framed audio windows and sample rate.

	Returns a list of (window_audio, sr).
	"""
	win = int(window_seconds * sr)
	hop = int(hop_seconds * sr)
	frames = []
	for start in range(0, max(1, len(audio) - win + 1), hop):
		end = start + win
		if end > len(audio):
			break
		frames.append((audio[start:end], sr))
	return frames

