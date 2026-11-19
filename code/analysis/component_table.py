"""Write the LaTeX table of a CP model's components (tensor4gmri.component_table) to --output:
each component's weight and share of the data's sum of squares, their sum and the model fit,
read from the model's model.h5. A booktabs tabular without caption, to \\input in a table."""

import argparse
from pathlib import Path

from tensor4gmri.component_table import component_table, read_component_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="the model directory (cp_rank_<r>/, as gmri's --model) or its model.h5",
    )
    parser.add_argument("--output", type=Path, required=True, help="the .tex to write")
    args = parser.parse_args()

    model = args.model / "model.h5" if args.model.is_dir() else args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(component_table(read_component_metrics(model)))
    print(f"Wrote {args.output}")
