"""Print what a fit of one subject group needs, on one line of stdout (the stages capture it):

- `--print exclude` (default): the subjects of the data not in the group, for
  `gmri --exclude-subjects` (an empty line for `all`)
- `--print stratify`: the subject-info column the replicability halves are balanced on
- `--print covariates`: the categorical primary variables of the subject statistics
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tensor4gmri.config import GROUPS, PRIMARY_CATEGORICAL, STRATIFY_BY
from tensor4gmri.subject_info import excluded_for_group, read_subject_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument(
        "--print",
        dest="what",
        default="exclude",
        choices=("exclude", "stratify", "covariates"),
    )
    parser.add_argument(
        "--subject-info",
        type=Path,
        help="subject_info.csv (for exclude)",
    )
    parser.add_argument("--input", type=Path, help="roi_signal.parquet (for exclude)")
    args = parser.parse_args()

    if args.what == "stratify":
        print(STRATIFY_BY[args.group])
    elif args.what == "covariates":
        print(" ".join(PRIMARY_CATEGORICAL[args.group]))
    else:
        if args.subject_info is None or args.input is None:
            parser.error("--print exclude needs --subject-info and --input")
        present = (
            pq.read_table(args.input, columns=["subject"])["subject"]
            .unique()
            .to_pylist()
        )
        try:
            excluded = excluded_for_group(
                read_subject_info(args.subject_info),
                args.group,
                present,
            )
        except ValueError as error:
            sys.exit(f"{args.subject_info}: {error}")
        print(
            f"group {args.group}: {len(present) - len(excluded)} of {len(present)} subjects",
            file=sys.stderr,
        )
        print(" ".join(excluded))
