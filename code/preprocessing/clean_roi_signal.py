"""Write the copy of gmri preprocess's roi_signal.parquet that the fits read, and the same table
as csv next to it (<output>.csv), cleaned in three steps (tensor4gmri.roi_signal):

1. remove every subject's ROI with fewer than --min-voxels voxels (all its time points);
2. remove the rows with NaN values;
3. remove, for every subject, each label that some subject has no row of left.

The copy keeps the input's columns and file metadata (which gmri reads) and records min_voxels.
Writes every removed entry with its reason to --report and the labels of step 3, with the
subjects without them, to --label-report.
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tensor4gmri.roi_signal import clean_roi_signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="roi_signal.parquet")
    parser.add_argument(
        "--min-voxels",
        type=int,
        required=True,
        help="Smallest ROI kept (0: any size)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="The parquet to write",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="csv of the removed entries",
    )
    parser.add_argument(
        "--label-report",
        type=Path,
        required=True,
        help="csv of the labels removed for every subject",
    )
    args = parser.parse_args()

    table = pq.read_table(args.input)
    frame = table.to_pandas()
    kept, entries, labels = clean_roi_signal(frame, args.min_voxels)
    metadata = {
        **(table.schema.metadata or {}),
        b"min_voxels": str(args.min_voxels).encode(),
    }
    output = pa.Table.from_pandas(kept, schema=table.schema, preserve_index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output.replace_schema_metadata(metadata), args.output)
    kept.to_csv(args.output.with_suffix(".csv"), index=False)
    entries.to_csv(args.report, index=False)
    labels.to_csv(args.label_report, index=False)

    n_entries = len(frame[["subject", "label"]].drop_duplicates())
    if args.min_voxels > 0:
        n_small = int((entries["reason"] == "small").sum())
        print(
            f"{n_small} subject x label entries (of {n_entries}) with fewer than "
            f"{args.min_voxels} voxels removed",
        )
    n_nan = sum(
        len(t.split()) for t in entries.loc[entries["reason"] == "nan", "time_points"]
    )
    print(
        f"{n_nan} more row{'' if n_nan == 1 else 's'} (subject, time point, label) with NaN "
        "values removed",
    )
    if not labels.empty:
        print(
            f"{len(labels)} label{'' if len(labels) == 1 else 's'} some subject lacks after that, "
            f"removed for every subject: {', '.join(map(str, labels['label']))}",
        )
    print(f"{kept['label'].nunique()} of {frame['label'].nunique()} labels kept")
    print(f"Removed entries in {args.report}, removed labels in {args.label_report}")
    print(f"Written {args.output} and {args.output.with_suffix('.csv')}")
