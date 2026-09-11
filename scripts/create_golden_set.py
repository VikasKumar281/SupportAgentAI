import argparse
from pathlib import Path

import pandas as pd


INTENTS = [
    "order_delivery",
    "returns_refunds",
    "payment_billing",
    "account_security",
    "subscription_prime",
    "digital_content",
    "device_technical",
    "product_order_issue",
    "customer_service_complaint",
    "other_non_actionable",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/intent_discovery_labeled_draft.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/golden_set.csv"
    )
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        "sample_id",
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response",
        "final_intent",
        "label_confidence",
        "label_notes",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df["final_intent"].isin(INTENTS)
        & df["customer_text"].fillna("").str.strip().ne("")
    ].copy()

    selected_parts = []

    counts = df["final_intent"].value_counts()

    base_per_intent = max(10, args.n // len(INTENTS))

    for intent in INTENTS:
        intent_df = df[df["final_intent"] == intent].copy()

        if intent_df.empty:
            continue

        take = min(base_per_intent, len(intent_df))

        selected_parts.append(
            intent_df.sample(
                n=take,
                random_state=42
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    remaining = df.loc[~df["sample_id"].isin(selected["sample_id"])].copy()

    remaining_needed = args.n - len(selected)

    if remaining_needed > 0:
        extra = remaining.sample(
            n=min(remaining_needed, len(remaining)),
            random_state=42
        )
        selected = pd.concat([selected, extra], ignore_index=True)

    selected = selected.sample(
        n=min(args.n, len(selected)),
        random_state=42
    ).reset_index(drop=True)

    selected["gold_intent"] = selected["final_intent"]
    selected["gold_escalate"] = ""
    selected["gold_escalation_reason"] = ""
    selected["gold_label_confidence"] = ""
    selected["gold_label_notes"] = ""
    selected["human_review_status"] = "pending"

    columns = [
        "sample_id",
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response",
        "gold_intent",
        "gold_escalate",
        "gold_escalation_reason",
        "gold_label_confidence",
        "gold_label_notes",
        "human_review_status",
    ]

    golden = selected[columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(output_path, index=False)

    print(f"Source rows: {len(df)}")
    print(f"Golden rows: {len(golden)}")
    print(f"Output: {output_path}")
    print("\nGolden set draft distribution:")
    print(golden["gold_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()