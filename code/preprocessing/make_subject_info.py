"""Write subject_info.csv, the study's subjects: those of the labels not marked `exclude` and
diagnosed REF or iNPH, keyed by `subjects`, with Group and Sex named and Stratum their
combination. Prints how many there are per Group and Sex, and which ids are left out and why."""

import argparse
from pathlib import Path

import pandas as pd
from tensor4gmri.subject_info import (
    left_out_labels,
    read_labels,
    subject_info_from_labels,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        type=Path,
        help="Subject labels csv ($SUBJECT_LABELS)",
    )
    parser.add_argument("output_dir", type=Path, help="Where to write subject_info.csv")
    args = parser.parse_args()

    labels = read_labels(args.labels)
    info = subject_info_from_labels(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info.to_csv(args.output_dir / "subject_info.csv", index=False)
    print(f"{len(info)} of {len(labels)} subjects in {args.labels}:")
    print(pd.crosstab(info["Group"], info["Sex"], margins=True).to_string())
    for reason, subjects in left_out_labels(labels).items():
        if subjects:
            print(f"Left out, {reason}: {', '.join(subjects)}")
