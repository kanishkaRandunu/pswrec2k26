"""
Download RecBole datasets directly (avoids torch.distributed barrier issue).
Saves to scratch by default: /scratch/up63/kd6504/recbole_datasets/

Usage:
  python scripts/download_datasets_recobole.py
  python scripts/download_datasets_recobole.py --output-dir dataset --datasets steam-duprem
"""
import argparse
import os
import urllib.request as ur
import zipfile
import yaml

DATA_PATH = "/scratch/up63/kd6504/recbole_datasets/"
URL_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RecBole", "recbole", "properties", "dataset", "url.yaml"
)

# steam-duprem = Steam (duplicate removal) from RecSysDatasets; uses RecBole steam-merged URL
STEAM_DUPREM_URL = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Steam/merged/steam.zip"


def download_url(url, folder):
    os.makedirs(folder, exist_ok=True)
    filename = url.rpartition("/")[2]
    path = os.path.join(folder, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  Using existing {filename}")
        return path
    print(f"  Downloading {url}...")
    ur.urlretrieve(url, path)
    return path


def extract_and_rename(zip_path, dest_folder, dataset_name):
    with zipfile.ZipFile(zip_path, "r") as f:
        zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
        f.extractall(dest_folder)

    # Handle case where zip extracts to a subfolder
    subdir = os.path.join(dest_folder, zip_basename)
    if os.path.isdir(subdir):
        for f in os.listdir(subdir):
            src = os.path.join(subdir, f)
            dst = os.path.join(dest_folder, f)
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
        os.rmdir(subdir)

    # Rename atomic files to match dataset name (e.g. steam.inter -> steam-duprem.inter)
    for f in os.listdir(dest_folder):
        if zip_basename in f and f.endswith((".inter", ".user", ".item")):
            new_name = f.replace(zip_basename, dataset_name)
            os.rename(os.path.join(dest_folder, f), os.path.join(dest_folder, new_name))

    os.remove(zip_path)


def main():
    parser = argparse.ArgumentParser(description="Download RecBole datasets.")
    parser.add_argument("--output-dir", "-o", type=str, default=DATA_PATH,
                        help=f"Output base directory (default: {DATA_PATH})")
    parser.add_argument("--datasets", "-d", type=str, default=None, nargs="+",
                        help="Datasets to download (default: amazon-beauty, ml-1m, steam-not-merged, steam-duprem)")
    args = parser.parse_args()

    output_base = os.path.abspath(args.output_dir)

    # Load URLs from RecBole's url.yaml (no recbole import to avoid numpy/torch issues)
    url_path = os.path.abspath(URL_YAML)
    if os.path.exists(url_path):
        with open(url_path) as f:
            urls = yaml.safe_load(f)
    else:
        urls = {}
    urls["steam-duprem"] = STEAM_DUPREM_URL

    datasets_td = args.datasets or ["amazon-beauty", "ml-1m", "steam-not-merged", "steam-duprem"]

    for dataset_name in datasets_td:
        if dataset_name not in urls:
            print(f"Skipping {dataset_name}: URL not found")
            continue
        dest_folder = os.path.join(output_base, dataset_name)
        if os.path.exists(os.path.join(dest_folder, f"{dataset_name}.inter")):
            print(f"Already exists: {dataset_name}")
            continue
        print(f"Downloading {dataset_name}...")
        zip_path = download_url(urls[dataset_name], dest_folder)
        extract_and_rename(zip_path, dest_folder, dataset_name)
        print(f"Done: {dataset_name}")


if __name__ == "__main__":
    main()
