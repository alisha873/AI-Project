import json
from pathlib import Path

results_path = Path("evaluation/results/gnn_baseline_results.json")
results = json.loads(results_path.read_text(encoding=" utf-8\))

actual = [item["actual"] for item in results["test"]["predictions"]]
predicted = [item["predicted"] for item in results["test"]["predictions"]]

def metrics(y_true, y_pred):
    tp = sum(a == 1 and p == 1 for a, p in zip(y_true, y_pred))
    fp = sum(a == 0 and p == 1 for a, p in zip(y_true, y_pred))
    fn = sum(a == 1 and p == 0 for a, p in zip(y_true, y_pred))
    tn = sum(a == 0 and p == 0 for a, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

majority_label = max(set(actual), key=actual.count)
baseline = [majority_label] * len(actual)

print(json.dumps({
    "test_examples": len(actual),
    "actual_labels": actual,
    "gnn": metrics(actual, predicted),
    "majority_class_baseline": {
        "prediction": majority_label,
        **metrics(actual, baseline),
    },
    "caveat": (
        "Only four test examples. These edge labels have not been "
        "independently validated as successful attack transitions."
    ),
}, indent=2))
