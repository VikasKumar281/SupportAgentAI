import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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
        "--output-dir",
        default="data/processed"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    df = df[
        df["final_intent"].isin(INTENTS)
        & df["customer_text"].fillna("").str.strip().ne("")
    ].copy()

    df = df.drop_duplicates(
        subset=["customer_text"]
    ).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["final_intent"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["final_intent"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )

    print(f"Total labeled examples: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test: {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["final_intent"].value_counts().to_string())

    print("\nValidation distribution:")
    print(validation_df["final_intent"].value_counts().to_string())

    print("\nTest distribution:")
    print(test_df["final_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()