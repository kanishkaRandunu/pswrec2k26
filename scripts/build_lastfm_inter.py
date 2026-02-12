#!/usr/bin/env python3
"""
Build a RecBole-compatible lastfm.inter file with timestamps.

The standard RecSysDatasets conversion uses user_artists.dat which has NO
timestamps (just listen counts).  For sequential recommendation we need
temporal ordering, so this script uses user_taggedartists-timestamps.dat
instead:

    userID \t artistID \t tagID \t timestamp   (ms epoch)

Processing steps:
  1. Download and extract hetrec2011-lastfm-2k.zip (if not cached).
  2. Read user_taggedartists-timestamps.dat.
  3. Deduplicate by (userID, artistID), keeping the earliest timestamp.
  4. Write lastfm.inter with header:
       user_id:token \t artist_id:token \t timestamp:float

Usage:
  python scripts/build_lastfm_inter.py /scratch/up63/kd6504/recbole_datasets/lastfm
"""
import argparse
import csv
import io
import os
import tempfile
import urllib.request
import zipfile
from collections import defaultdict

LASTFM_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
TIMESTAMPS_FILE = "user_taggedartists-timestamps.dat"


def download_and_extract(cache_dir: str) -> str:
    """Download the zip if needed, extract, and return path to extracted dir."""
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "hetrec2011-lastfm-2k.zip")
    extract_dir = os.path.join(cache_dir, "hetrec2011-lastfm-2k")

    if not os.path.isfile(zip_path):
        print(f"Downloading {LASTFM_URL} ...")
        urllib.request.urlretrieve(LASTFM_URL, zip_path)
        print(f"Saved to {zip_path}")
    else:
        print(f"Using cached zip: {zip_path}")

    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
    else:
        print(f"Using cached extraction: {extract_dir}")

    return extract_dir


def read_tagged_timestamps(extract_dir: str) -> dict:
    """Read user_taggedartists-timestamps.dat and return earliest timestamp per (user, artist)."""
    filepath = os.path.join(extract_dir, TIMESTAMPS_FILE)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    earliest = {}  # (userID, artistID) -> earliest_timestamp

    with open(filepath, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            user_id = parts[0]
            artist_id = parts[1]
            # tagID = parts[2]  # not needed
            timestamp = float(parts[3])

            key = (user_id, artist_id)
            if key not in earliest or timestamp < earliest[key]:
                earliest[key] = timestamp

    return earliest


def write_inter(earliest: dict, output_path: str):
    """Write RecBole .inter file sorted by (user_id, timestamp)."""
    # Convert to list and sort by user then timestamp
    rows = [(uid, aid, ts) for (uid, aid), ts in earliest.items()]
    rows.sort(key=lambda r: (int(r[0]), r[2]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        f.write("user_id:token\tartist_id:token\ttimestamp:float\n")
        for uid, aid, ts in rows:
            # Convert ms epoch to seconds for consistency
            f.write(f"{uid}\t{aid}\t{ts / 1000.0:.1f}\n")

    print(f"Wrote {output_path} ({len(rows)} interactions)")


def main():
    parser = argparse.ArgumentParser(
        description="Build lastfm.inter with timestamps for sequential recommendation."
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write lastfm.inter (e.g. /scratch/up63/kd6504/recbole_datasets/lastfm)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache the downloaded zip. Defaults to a temp directory.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="lastfm_build_")
    extract_dir = download_and_extract(cache_dir)
    earliest = read_tagged_timestamps(extract_dir)

    output_path = os.path.join(args.output_dir, "lastfm.inter")
    write_inter(earliest, output_path)

    print(f"\nDataset stats:")
    users = set(uid for uid, _ in earliest.keys())
    artists = set(aid for _, aid in earliest.keys())
    print(f"  Users:        {len(users)}")
    print(f"  Artists:      {len(artists)}")
    print(f"  Interactions: {len(earliest)}")


if __name__ == "__main__":
    main()
