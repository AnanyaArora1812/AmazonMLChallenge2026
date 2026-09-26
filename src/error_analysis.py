import pandas as pd


def find_errors(data):
    false_positives = data[
        (data["actual"] == 0) &
        (data["predicted"] == 1)
    ]

    false_negatives = data[
        (data["actual"] == 1) &
        (data["predicted"] == 0)
    ]

    return false_positives, false_negatives


def save_errors(false_positives, false_negatives):
    false_positives.to_csv(
        "outputs/false_positives.csv",
        index=False
    )

    false_negatives.to_csv(
        "outputs/false_negatives.csv",
        index=False
    )