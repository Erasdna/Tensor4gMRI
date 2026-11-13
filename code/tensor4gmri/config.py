"""Constants shared by the pipeline: time points, label codes and the subject groups."""

from typing import Literal, get_args

# Hours after injection of the scans in the tensor; each subject's scan closest to each.
TIME_POINTS: tuple[int, ...] = (2, 4, 8, 24)

# Codes of the subject labels CSV ($SUBJECT_LABELS).
GROUP_LABELS: dict[int, str] = {0: "REF", 1: "iNPH"}
SEX_LABELS: dict[int, str] = {0: "M", 1: "F"}

# The subjects a model is fitted on: every subject, one diagnosis or one sex.
Group = Literal["all", "REF", "iNPH", "M", "F"]
GROUPS: tuple[Group, ...] = get_args(Group)

# Subject-info (column, value) of each group's subjects; None: every subject.
GROUP_FILTER: dict[Group, tuple[str, str] | None] = {
    "all": None,
    "REF": ("Group", "REF"),
    "iNPH": ("Group", "iNPH"),
    "M": ("Sex", "M"),
    "F": ("Sex", "F"),
}

# Subject-info column the replicability halves are balanced on: what varies within the group.
# Stratum is <Group>_<Sex>, i.e. both.
STRATIFY_BY: dict[Group, str] = {
    "all": "Stratum",
    "REF": "Sex",
    "iNPH": "Sex",
    "M": "Group",
    "F": "Group",
}

# Categorical primary variables of the subject statistics (gmri stats): those that vary within
# the group.
PRIMARY_CATEGORICAL: dict[Group, tuple[str, ...]] = {
    "all": ("Group", "Sex"),
    "REF": ("Sex",),
    "iNPH": ("Sex",),
    "M": ("Group",),
    "F": ("Group",),
}
