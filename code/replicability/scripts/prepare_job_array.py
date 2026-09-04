import pandas as pd
from itertools import product
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration file for job array")
    parser.add_argument("config", type=Path, help="Path to configuration file")
    args = parser.parse_args()
    groups = ["M", "F", "REF", "iNPH"]
    ranks = [2, 3, 4, 5, 6]
    methods = ["CV", "half-half"]

    combinations = list(product(groups, ranks, methods))
    df = pd.DataFrame(combinations, columns=["Groups", "Ranks", "Methods"])
    df.to_csv(args.config, header=False, index=False)
