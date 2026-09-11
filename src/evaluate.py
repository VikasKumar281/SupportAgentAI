"""
Automated metrics: intent classification (accuracy, macro-F1, per-class,
confusion matrix) and escalation decision (precision/recall/F1, plus the
safety-critical false-negative rate: risky messages the system wrongly
auto-handled).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from src.config import INTENTS


def classification_metrics(y_true, y_pred, labels=None):
    labels = labels or INTENTS
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "per_class_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }


def escalation_metrics(y_true_escalate, y_pred_escalate):
    y_true_escalate = np.array(y_true_escalate, dtype=bool)
    y_pred_escalate = np.array(y_pred_escalate, dtype=bool)

    precision = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
    recall = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
    f1 = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)

    # Safety-critical: of messages a human reviewer WOULD have escalated,
    # how many did the system auto-handle instead? (false negatives, i.e.
    # the dangerous error direction). This is reported separately from F1
    # because F1 can look fine while hiding a bad FN rate if the escalate
    # class is small.
    should_escalate = y_true_escalate
    missed = should_escalate & (~y_pred_escalate)
    fn_rate = float(missed.sum() / should_escalate.sum()) if should_escalate.sum() > 0 else 0.0

    over_escalated = (~y_true_escalate) & y_pred_escalate
    fp_rate = float(over_escalated.sum() / (~y_true_escalate).sum()) if (~y_true_escalate).sum() > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "unsafe_auto_handle_rate": round(fn_rate, 4),   # should-escalate but didn't -> RISK
        "unnecessary_escalation_rate": round(fp_rate, 4),  # escalated but didn't need to -> COST
        "n_should_escalate": int(should_escalate.sum()),
        "n_total": int(len(y_true_escalate)),
    }


def save_metrics(metrics: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
