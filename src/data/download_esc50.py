import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"


def download_file(url: str, dest_path: Path) -> None:
	dest_path.parent.mkdir(parents=True, exist_ok=True)
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		total = int(r.headers.get("Content-Length", 0))
		with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)
					pbar.update(len(chunk))


def main() -> None:
	root = Path(__file__).resolve().parents[2]
	raw_dir = root / "data" / "raw"
	zip_path = raw_dir / "ESC-50-master.zip"
	data_root = raw_dir / "ESC-50-master"

	if not zip_path.exists():
		print("Downloading ESC-50...")
		download_file(ESC50_URL, zip_path)
	else:
		print("ESC-50 archive already exists.")

	if not data_root.exists():
		print("Extracting ESC-50...")
		with zipfile.ZipFile(zip_path, 'r') as z:
			z.extractall(raw_dir)
	else:
		print("ESC-50 already extracted.")

	print("Done. Files in:", data_root)


if __name__ == "__main__":
	main()

