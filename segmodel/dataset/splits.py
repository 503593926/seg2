"""HER2ST 患者留出: 标注切片评估分割, 同患者全部切片评估表达。"""

import re


def patient_id(name):
    match = re.fullmatch(r"([A-H])\d+", str(name))
    if match is None:
        raise ValueError(f"Expected a HER2ST slide name such as C1, got {name!r}")
    return match[1]


def patient_folds(labeled_names, expression_names):
    labeled_names, expression_names = list(labeled_names), list(expression_names)
    if len(expression_names) != len(set(expression_names)):
        raise ValueError("expression_names contains duplicate slides")
    missing = set(labeled_names) - set(expression_names)
    if missing:
        raise ValueError(f"Annotated slides missing from expression_names: {sorted(missing)}")
    patients = list(dict.fromkeys(patient_id(name) for name in labeled_names))
    return [
        {
            "train": [name for name in expression_names if patient_id(name) != patient],
            "test": [name for name in labeled_names if patient_id(name) == patient],
            "test_expression": [name for name in expression_names if patient_id(name) == patient],
            "test_patient": patient,
        }
        for patient in patients
    ]
