"""Write `<output-dir>/statistics/<model>__subject_stats_components.csv`: the correlation of each
pair of a model's components' subject coefficients, adjusted for the primary variables (partial r
with 95% CI, Freedman-Lane permutation p, FDR over the pairs) and unadjusted
(plot_stats.components.component_correlations)."""

import argparse
from pathlib import Path

from plot_stats.components import component_correlations
from plot_stats.data import model_stem, subject_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="The model of gmri decompose",
    )
    parser.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="subject_info.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="gmri stats' --output-dir",
    )
    parser.add_argument("--primary-categorical", nargs="*", default=[])
    parser.add_argument("--primary-numerical", nargs="*", default=[])
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    frame = subject_frame(args.model, args.subject_info)
    pairs = component_correlations(
        frame,
        args.primary_categorical,
        args.primary_numerical,
        args.n_permutations,
        args.seed,
        args.alpha,
    )
    folder = args.output_dir / "statistics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model_stem(args.model)}__subject_stats_components.csv"
    pairs.to_csv(path, index=False)
    print(
        f"{len(pairs)} pairs of components, {int(pairs['significant'].sum())} FDR-significant",
    )
    print(f"Written {path}")
