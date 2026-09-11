from pathlib import Path
import pandas as pd
import re

INPUT_PATH = Path("data/processed/amazonhelp_conversations.csv")
OUTPUT_DIR = Path("data/processed")

SAMPLE_SIZE = 500
RANDOM_STATE = 42


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
        dtype="string"
    )

    df = df[
        df["customer_text"].notna()
        & df["brand_response"].notna()
    ].copy()

    df["customer_text"] = df["customer_text"].str.strip()
    df["brand_response"] = df["brand_response"].str.strip()

    df = df[
        (df["customer_text"].str.len() >= 10)
        & (df["brand_response"].str.len() >= 10)
    ].copy()

    df["normalized_text"] = df["customer_text"].map(normalize_text)

    df = df[df["normalized_text"].str.len() >= 10]

    df = df.drop_duplicates(
        subset=["normalized_text"]
    )

    df["text_length"] = df["customer_text"].str.len()

    short = df[df["text_length"] <= 80]
    medium = df[
        (df["text_length"] > 80)
        & (df["text_length"] <= 200)
    ]
    long = df[df["text_length"] > 200]

    n_short = min(170, len(short))
    n_medium = min(170, len(medium))
    n_long = min(160, len(long))

    samples = []

    if n_short:
        samples.append(
            short.sample(
                n=n_short,
                random_state=RANDOM_STATE
            )
        )

    if n_medium:
        samples.append(
            medium.sample(
                n=n_medium,
                random_state=RANDOM_STATE + 1
            )
        )

    if n_long:
        samples.append(
            long.sample(
                n=n_long,
                random_state=RANDOM_STATE + 2
            )
        )

    sample = pd.concat(samples, ignore_index=True)

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    sample.insert(
        0,
        "sample_id",
        range(1, len(sample) + 1)
    )

    sample["candidate_intent"] = ""
    sample["final_intent"] = ""
    sample["label_confidence"] = ""
    sample["label_notes"] = ""

    columns = [
        "sample_id",
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response",
        "candidate_intent",
        "final_intent",
        "label_confidence",
        "label_notes"
    ]

    sample = sample[columns]

    output_path = OUTPUT_DIR / "intent_discovery_sample.csv"

    sample.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 70)
    print("INTENT DISCOVERY SAMPLE")
    print("=" * 70)
    print(f"Original pairs: {len(df):,}")
    print(f"Sample created: {len(sample):,}")
    print(f"Output: {output_path}")
    print()
    print(sample.head(10).to_string(index=False))


if __name__ == "__main__":
    main()