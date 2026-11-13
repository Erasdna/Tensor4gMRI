"""Write subject_info.csv: the labels of the subjects find_files.py found, keyed by
`subjects`, with Group and Sex named and Stratum their combination."""

import argparse
from pathlib import Path

import pandas as pd
from tensor4gmri.subject_info import read_labels, subject_info_from_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        type=Path,
        help="Subject labels csv ($SUBJECT_LABELS)",
    )
    parser.add_argument(
        "file_list",
        type=Path,
        help="files_to_process.csv from find_files.py",
    )
    parser.add_argument("output_dir", type=Path, help="Where to write subject_info.csv")
    args = parser.parse_args()

    subjects = pd.read_csv(args.file_list, dtype={"subject": str})["subject"].unique()
    info = subject_info_from_labels(read_labels(args.labels), subjects)
    info.to_csv(args.output_dir / "subject_info.csv", index=False)
    print(pd.crosstab(info["Group"], info["Sex"], margins=True).to_string())
