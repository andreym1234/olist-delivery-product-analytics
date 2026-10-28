"""Download the original public Olist archive; no private Kaggle token required."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = 'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce'
DOWNLOAD_URL = 'https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce'
EXPECTED_ARCHIVE_SHA256 = '967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a ZIP downloaded manually from Kaggle')
    parser.add_argument('--accept-new-version', action='store_true',
                        help='Explicitly allow a changed source archive; results may change')
    args = parser.parse_args()
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / 'olist.zip'
            print('Downloading public Olist dataset...')
            try:
                request = urllib.request.Request(DOWNLOAD_URL, headers={'User-Agent': 'olist-portfolio/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as output:
                    shutil.copyfileobj(response, output)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise SystemExit(f'Download failed: {type(exc).__name__}. '
                                 f'Download ZIP from {DATASET_URL} and run '
                                 'python scripts/download_data.py --archive path/to/archive.zip') from exc
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != EXPECTED_ARCHIVE_SHA256 and not args.accept_new_version:
            # ZIP metadata can change even when CSV content is unchanged.
            # Fall back to the shipped per-file manifest, when available.
            manifest_path = ROOT / 'data/source_manifest.json'
            matches = False
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())['files']
                with zipfile.ZipFile(archive) as z:
                    matches = all(hashlib.sha256(z.read(x['file'])).hexdigest() == x['sha256']
                                  for x in manifest)
            if not matches:
                raise SystemExit('Dataset version differs from the checked source. '
                                 'Review it, then rerun with --accept-new-version if intentional.')
        with zipfile.ZipFile(archive) as z:
            csvs = [m for m in z.infolist() if m.filename.endswith('.csv')]
            for member in csvs:
                if Path(member.filename).name != member.filename:
                    raise SystemExit('Unexpected nested filename in archive')
                with z.open(member) as source, (raw/member.filename).open('wb') as target:
                    shutil.copyfileobj(source, target)
        print(f'Ready: {len(csvs)} CSV files in data/raw. Source sha256: {digest}')


if __name__ == '__main__':
    main()
