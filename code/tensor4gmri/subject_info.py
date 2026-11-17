"""The subject table of the pipeline (subject_info.csv) and the subjects of each group."""

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from tensor4gmri.config import GROUP_FILTER, GROUP_LABELS, GROUPS, SEX_LABELS


def read_labels(path: Path) -> pd.DataFrame:
    """$SUBJECT_LABELS, with the subject ids as text. Raises `ValueError` if a `new_id` is
    repeated."""
    labels = pd.read_csv(path, dtype={"new_id": str, "old_id": str})
    repeated = sorted(set(labels.loc[labels["new_id"].duplicated(), "new_id"]))
    if repeated:
        raise ValueError(f"Subjects more than once in {path}: {repeated}")
    return labels


def _included(labels: pd.DataFrame) -> pd.Series:
    return labels["exclude"] == 0


def _diagnosed(labels: pd.DataFrame) -> pd.Series:
    return labels["Group"].isin(list(GROUP_LABELS))


def valid_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """The rows of `labels` of the study's subjects: not marked `exclude` and diagnosed REF or
    iNPH (a Group code of GROUP_LABELS). A missing value in either column is not valid.
    """
    return labels[_included(labels) & _diagnosed(labels)]


def left_out_labels(labels: pd.DataFrame) -> dict[str, list[str]]:
    """The ids of `labels` that valid_labels leaves out, by reason: "excluded" (`exclude` not
    0) and, of the others, "not REF or iNPH" (another or no Group code)."""
    included = _included(labels)
    return {
        "excluded": labels.loc[~included, "new_id"].tolist(),
        "not REF or iNPH": labels.loc[
            included & ~_diagnosed(labels),
            "new_id",
        ].tolist(),
    }


def subject_info_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """The valid rows of `labels` (valid_labels), keyed by `subjects` as gmri expects, with
    Group and Sex named (REF/iNPH, M/F) and Stratum their combination, e.g. REF_M.

    Raises `ValueError` for a Sex code without a name.
    """
    info = valid_labels(labels).rename(columns={"new_id": "subjects"})
    for column, names in (("Group", GROUP_LABELS), ("Sex", SEX_LABELS)):
        named = info[column].map(names)
        if named.isna().any():
            codes = sorted(set(info.loc[named.isna(), column]))
            raise ValueError(f"{column} codes without a name: {codes} (known: {names})")
        info[column] = named
    info["Stratum"] = info["Group"] + "_" + info["Sex"]
    return info.reset_index(drop=True)


def read_subject_info(path: Path) -> pd.DataFrame:
    """subject_info.csv with every cell as text."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def group_members(info: pd.DataFrame, group: str) -> list[str]:
    """The subjects of `info` in `group` (config.GROUP_FILTER), in table order."""
    if group not in GROUPS:
        raise ValueError(
            f"Unknown group '{group}' (expected one of {', '.join(GROUPS)})",
        )
    selection = GROUP_FILTER[group]  # type: ignore[index]
    if selection is None:
        return info["subjects"].tolist()
    column, value = selection
    return info.loc[info[column] == value, "subjects"].tolist()


def excluded_for_group(
    info: pd.DataFrame,
    group: str,
    present: Collection[str],
) -> list[str]:
    """The subjects of `present` (those in the data) not in `group`, sorted: what a fit of
    `group` leaves out with `--exclude-subjects`.

    Raises `ValueError` for an unknown group or a subject of `present` not in `info`.
    """
    unknown = sorted(set(present) - set(info["subjects"]))
    if unknown:
        raise ValueError(f"Subjects of the data not in the subject info: {unknown}")
    members = set(group_members(info, group))
    return sorted(s for s in present if s not in members)
