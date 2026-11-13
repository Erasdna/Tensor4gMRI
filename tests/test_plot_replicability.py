import numpy as np
import pandas as pd
from analysis.plot_replicability import fms_summary


def test_fms_summary_per_rank_counts_failed_splits() -> None:
    fms = pd.DataFrame(
        {"rank": [2, 2, 2, 3, 3, 3], "fms": [0.9, 0.95, 1.0, 0.5, np.nan, 0.7]},
    )
    summary = fms_summary(fms).set_index("rank")
    assert summary.loc[2, "median"] == 0.95
    assert summary.loc[3, "median"] == 0.6
    assert summary["n_splits"].tolist() == [3, 2]
    assert summary["n_failed"].tolist() == [0, 1]
