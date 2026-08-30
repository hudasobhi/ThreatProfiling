"""
download.py — Downloads CMU CERT r6.2 dataset from KiltHub.

Usage:
    python data/download.py --token YOUR_KILTHUB_TOKEN --output data/raw/

The dataset is publicly available at:
    https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
"""

import argparse
import os
import urllib.request


CERT_FILES = {
    "logon.csv":        "https://kilthub.cmu.edu/ndownloader/files/logon",
    "device.csv":       "https://kilthub.cmu.edu/ndownloader/files/device",
    "email.csv":        "https://kilthub.cmu.edu/ndownloader/files/email",
    "file.csv":         "https://kilthub.cmu.edu/ndownloader/files/file",
    "http.csv":         "https://kilthub.cmu.edu/ndownloader/files/http",
    "psychometric.csv": "https://kilthub.cmu.edu/ndownloader/files/psychometric",
}


def download_file(url: str, dest: str, token: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    request = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
    print(f"Downloading {os.path.basename(dest)} ...")
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        out.write(response.read())
    print(f"  Saved to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download CMU CERT r6.2 dataset.")
    parser.add_argument("--token",  type=str, required=True, help="KiltHub API token.")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output directory.")
    args = parser.parse_args()

    for filename, url in CERT_FILES.items():
        dest = os.path.join(args.output, filename)
        if os.path.exists(dest):
            print(f"  {filename} already exists, skipping.")
            continue
        download_file(url, dest, args.token)

    print("\nAll files downloaded. Run data/preprocess.py to generate processed partitions.")


if __name__ == "__main__":
    main()
