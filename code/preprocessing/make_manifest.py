"""Write manifest.csv, the input of `gmri preprocess`, from files_to_process.csv."""

import argparse
from pathlib import Path

import pandas as pd
from tensor4gmri.manifest import image_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file_list",
        type=Path,
        help="files_to_process.csv from find_files.py",
    )
    parser.add_argument("output_dir", type=Path, help="Where to write manifest.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list, dtype={"subject": str})
    manifest = image_manifest(file_list, args.file_list.parent)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"{len(manifest)} scans of {manifest['subject'].nunique()} subjects")
