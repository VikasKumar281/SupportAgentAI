import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "golden_set_review_ready.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "golden_set_human_review.csv"
)

df = pd.read_csv(INPUT_PATH)

review_df = pd.DataFrame({
    "sample_id": range(1, len(df) + 1),
    "customer_text": df["customer_text"].fillna(""),
    "brand_response": df["brand_response"].fillna(""),
    "gold_intent": "",
    "gold_escalate": "",
    "gold_escalation_reason": "",
    "review_notes": "",
    "review_status": "pending"
})

review_df.to_csv(OUTPUT_PATH, index=False)

print()
print("Human Review Set")
print("================")
print()
print(f"Examples: {len(review_df)}")
print(f"Saved to: {OUTPUT_PATH}")