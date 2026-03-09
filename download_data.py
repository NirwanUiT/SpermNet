#!/usr/bin/env python3
"""
download_data.py

Downloads the VISEM-Tracking dataset from Zenodo (record ID: 7293726).
- Lists all available files via the Zenodo REST API
- Prints filenames, sizes, and download links
- Downloads files into data/raw/ with a tqdm progress bar
- Handles errors gracefully

Usage:
    python download_data.py              # List files only (no download)
    python download_data.py --download   # List files and then download all
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm
from pathlib import Path

ZENODO_RECORD_ID = "7293726"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def fetch_record_metadata():
    """Fetch record metadata from the Zenodo REST API."""
    print(f"Fetching metadata from {ZENODO_API_URL} ...")
    try:
        resp = requests.get(ZENODO_API_URL, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError:
        print("ERROR: Could not connect to Zenodo. Check your internet connection.")
        sys.exit(1)
    except requests.Timeout:
        print("ERROR: Request to Zenodo timed out.")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR: Zenodo returned HTTP {resp.status_code}: {e}")
        sys.exit(1)

    return resp.json()


def list_files(metadata):
    """Parse and display the files available in the Zenodo record."""
    files = metadata.get("files", [])
    if not files:
        print("No files found in this Zenodo record.")
        return []

    print(f"\n{'='*80}")
    print(f"VISEM-Tracking Dataset  —  Zenodo Record {ZENODO_RECORD_ID}")
    print(f"Title: {metadata.get('metadata', {}).get('title', 'N/A')}")
    print(f"DOI:   {metadata.get('doi', 'N/A')}")
    print(f"{'='*80}")
    print(f"\n{'#':<4} {'Filename':<55} {'Size':>12}  Link")
    print("-" * 120)

    total_size = 0
    for idx, f in enumerate(files, 1):
        size_bytes = f.get("size", 0)
        total_size += size_bytes
        size_str = _human_size(size_bytes)
        name = f.get("key", "unknown")
        link = f.get("links", {}).get("self", "N/A")
        print(f"{idx:<4} {name:<55} {size_str:>12}  {link}")

    print("-" * 120)
    print(f"{'Total:':<60} {_human_size(total_size):>12}")
    print(f"{'Files:':<60} {len(files):>12}")
    print()

    return files


def download_file(url, dest_path, expected_size=None):
    """Download a single file with a tqdm progress bar."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"  ERROR: Connection failed for {dest_path.name}")
        return False
    except requests.Timeout:
        print(f"  ERROR: Download timed out for {dest_path.name}")
        return False
    except requests.HTTPError as e:
        print(f"  ERROR: HTTP {resp.status_code} for {dest_path.name}: {e}")
        return False

    total = expected_size or int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as fout, tqdm(
        desc=dest_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        ncols=100,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                fout.write(chunk)
                bar.update(len(chunk))

    return True


def download_all(files):
    """Download every file in the list into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} file(s) into {RAW_DIR.resolve()}\n")

    success, failed = 0, 0
    for f in files:
        name = f.get("key", "unknown")
        url = f.get("links", {}).get("self")
        size = f.get("size", 0)

        if not url:
            print(f"  SKIP: No download link for {name}")
            failed += 1
            continue

        dest = RAW_DIR / name

        # Skip if already downloaded with correct size
        if dest.exists() and dest.stat().st_size == size:
            print(f"  SKIP (already exists): {name}")
            success += 1
            continue

        ok = download_file(url, dest, expected_size=size)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed.")


def _human_size(nbytes):
    """Convert byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Download the VISEM-Tracking dataset from Zenodo."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files. Without this flag, only list files.",
    )
    args = parser.parse_args()

    metadata = fetch_record_metadata()
    files = list_files(metadata)

    if not files:
        return

    if args.download:
        download_all(files)
    else:
        print("Run with --download to start downloading.\n"
              "  python download_data.py --download")


if __name__ == "__main__":
    main()
