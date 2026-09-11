import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="data/processed/train.csv"
    )
    parser.add_argument(
        "--test",
        default="data/processed/test.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/majority_baseline_results.txt"
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)

    majority_intent = train_df["final_intent"].value_counts().idxmax()

    y_true = test_df["final_intent"]
    y_pred = [majority_intent] * len(test_df)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    output = f"""
Majority Baseline
=================

Majority Intent: {majority_intent}
Test Examples: {len(test_df)}
Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}

Classification Report:
{report}
"""

    print(output)

    Path(args.output).write_text(
        output,
        encoding="utf-8"
    )

    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()