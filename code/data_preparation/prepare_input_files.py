import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import nibabel as nib


def compute_time(filename: Path, n: int):
    t_raw = int(filename.stem[-10:-4])
    # Compute time in hours
    return (
        (t_raw % 100)
        + (t_raw // 100) % 100 * 60
        + (t_raw // 10000) % 100 * 3600
        + n * 24 * 3600
    ) / 3600


def find_images(folder):
    n = 0
    img_list = []
    # Find images from first day
    while len(img_list) == 0 and n < 10:
        n = n + 1
        img_list = list(folder.glob("*normalized-201?010" + str(n) + "*"))

    if len(img_list) == 0:
        raise FileNotFoundError

    time = []
    for img in img_list:
        time.append(compute_time(img, 0))
    # Add second day
    second_day = list(folder.glob("*normalized-201?010" + str(n + 1) + "*"))
    if len(second_day) < 1:
        raise FileNotFoundError
    img_list.append(second_day[0])
    time.append(compute_time(img_list[-1], 1))
    time = np.array(time)
    img_list = np.array(img_list)

    sorted_ids = np.argsort(time)
    img_list = img_list[sorted_ids]
    time = time[sorted_ids]

    reference = str(img_list[0].resolve().absolute())
    time = time - time[0]

    tracer = {"actual_time": [], "path": [], "time_point": []}
    prev = None
    # Find image closest to 2,4,6,8 and 24 hours after
    for time_point in [2, 4, 8, 24]:
        img_id = np.argmin(np.abs(time - time_point))
        tracer["path"].append(str(img_list[img_id].resolve().absolute()))
        tracer["actual_time"].append(time[img_id])
        tracer["time_point"].append(time_point)

        if prev is not None:
            if prev == str(img_list[img_id]):
                raise FileNotFoundError

        print(f"Image after {time_point} hours")
        print("     Actual time ", time[img_id], " filename ", img_list[img_id].name)

        prev = str(img_list[img_id])

    subj = folder.name
    sas = folder.joinpath(subj + "-sasparc.nii.gz").resolve().absolute()
    parenchyma = subj_path.joinpath(subj + "-refroi.nii.gz").resolve().absolute()
    for f in [sas, parenchyma]:
        if not f.exists():
            raise FileNotFoundError

    return reference, tracer, sas, parenchyma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tracer data and identify files")
    parser.add_argument("folder", type=Path, help="Path to folder we iterate over")
    parser.add_argument("labels", type=Path, help="Path to data labels")
    parser.add_argument("save", type=Path, help="Path to save file")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)

    tracer_path = Path(args.folder)

    save_df: dict = {
        "subject": [],
        "time_point": [],
        "baseline_path": [],
        "post_injection_path": [],
        "mask_path": [],
        "segmentation_path": [],
        "actual_time": [],
        "old_id": [],
    }
    for subj, old_id in zip(df["new_id"], df["old_id"]):
        subj_path = tracer_path.joinpath(subj)
        print(subj)
        try:
            reference, tracer, sas, parenchyma = find_images(subj_path)
        except FileNotFoundError:
            print("Missing files...skipping")
            continue

        # Merge
        parenchyma_nifti = nib.load(parenchyma)
        parenchyma_data = parenchyma_nifti.get_fdata()
        sas_data = nib.load(sas).get_fdata()
        seg = np.where(sas_data > 0, sas_data + 10000, parenchyma_data)
        segmentation_path = subj_path.joinpath(f"{subj}-segmentation.nii.gz")
        brainmask_path = subj_path.joinpath(f"{subj}-brainmask.nii.gz")
        nib.save(
            nib.Nifti1Image(seg, parenchyma_nifti.affine, parenchyma_nifti.header),
            segmentation_path,
        )
        nib.save(
            nib.Nifti1Image(seg > 0, parenchyma_nifti.affine, parenchyma_nifti.header),
            brainmask_path,
        )

        for time, path, actual_time in zip(
            tracer["time_point"],
            tracer["path"],
            tracer["actual_time"],
        ):
            save_df["subject"].append(subj)
            save_df["time_point"].append(time)
            save_df["baseline_path"].append(reference)
            save_df["post_injection_path"].append(str(path))
            save_df["mask_path"].append(brainmask_path)
            save_df["segmentation_path"].append(segmentation_path)
            save_df["actual_time"].append(actual_time)
            save_df["old_id"].append(old_id)

    pd.DataFrame(save_df).to_csv(args.save)
