import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "failure_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "top_failure_examples.csv"
)

df = pd.read_csv(INPUT_PATH, low_memory=False).fillna("")

intent_errors = df[
    df["error_type"] == "intent_misclassification"
].copy()

intent_errors["confidence_gap"] = (
    1 - intent_errors["intent_confidence"]
)

top_intent = (
    intent_errors
    .sort_values(
        ["confidence_gap", "retrieval_similarity"],
        ascending=[False, True]
    )
    .head(20)
)

escalation_errors = df[
    df["error_type"] == "escalation_error"
].copy()

top_escalation = (
    escalation_errors
    .sort_values(
        ["intent_confidence", "retrieval_similarity"],
        ascending=[True, True]
    )
    .head(20)
)

result = pd.concat(
    [
        top_intent,
        top_escalation
    ],
    ignore_index=True
)

result = result.drop_duplicates(
    subset=["sample_id"]
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Top Failure Examples")
print("====================")
print()
print(f"Examples selected: {len(result)}")
print()
print(result[
    [
        "sample_id",
        "gold_intent",
        "predicted_intent",
        "intent_confidence",
        "retrieval_similarity",
        "gold_escalate",
        "predicted_escalate"
    ]
].to_string(index=False))
print()
print(f"Saved to: {OUTPUT_PATH}")