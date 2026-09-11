import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "agent_evaluation_examples.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "failure_analysis.csv"
)

df = pd.read_csv(INPUT_PATH, low_memory=False).fillna("")

df["intent_error"] = (
    df["gold_intent"] != df["predicted_intent"]
)

df["escalation_error"] = (
    df["gold_escalate"].astype(str).str.lower()
    != df["predicted_escalate"].astype(str).str.lower()
)

intent_errors = df[df["intent_error"]].copy()

intent_errors["error_type"] = "intent_misclassification"

escalation_errors = df[df["escalation_error"]].copy()

escalation_errors["error_type"] = "escalation_error"

failures = pd.concat(
    [
        intent_errors,
        escalation_errors
    ],
    ignore_index=True
)

failures = failures.drop_duplicates(
    subset=["sample_id", "error_type"]
)

columns = [
    "sample_id",
    "customer_text",
    "gold_intent",
    "predicted_intent",
    "intent_confidence",
    "retrieval_similarity",
    "gold_escalate",
    "predicted_escalate",
    "gold_escalation_reason",
    "error_type"
]

failures = failures[columns]

failures.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Failure Analysis")
print("================")
print()
print(f"Total failures: {len(failures)}")
print(f"Intent errors: {len(intent_errors)}")
print(f"Escalation errors: {len(escalation_errors)}")
print()
print("Intent confusion pairs:")
print(
    intent_errors
    .groupby(["gold_intent", "predicted_intent"])
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_string()
)
print()
print(f"Saved to: {OUTPUT_PATH}")