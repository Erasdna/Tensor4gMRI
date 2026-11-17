import argparse
import numpy as np
import h5py
from pathlib import Path
from gMRItensor.roi_groups import VENTRICLE_IDS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make breakdown of regions included in CP model",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to h5py file with model weights",
    )
    args = parser.parse_args()

    with h5py.File(args.model, "r") as f:
        rois = f["rois"][()]

    print(rois)

    ventricles = np.isin(rois, VENTRICLE_IDS).sum()
    csf = (rois > 10000).sum()
    grey_matter = ((rois >= 1000) & (rois < 3000)).sum()
    white_matter = ((rois >= 3000) & (rois < 5000)).sum()

    all = ventricles + csf + grey_matter + white_matter
    rest = len(rois) - all

    print(
        ventricles,
        csf,
        grey_matter,
        white_matter,
        rest,
    )
