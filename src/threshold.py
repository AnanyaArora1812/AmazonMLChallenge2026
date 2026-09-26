import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_f05 = 0
    results = []

    for threshold in np.arange(0.50, 0.96, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f05 = fbeta_score(y_true, predictions, beta=0.5, zero_division=0)

        results.append({
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "f0.5": f05
        })

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    return best_threshold, best_f05, results