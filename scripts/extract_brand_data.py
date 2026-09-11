from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AMAZONHELP DATA EXTRACTION")
    print("=" * 70)
    print(f"Brand: {BRAND}")
    print(f"Input: {DATA_PATH}")
    print()


    brand_tweet_ids = set()
    parent_tweet_ids = set()

    total_rows = 0
    brand_rows = 0

    print("PASS 1: Finding AmazonHelp tweets...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            dtype={
                "tweet_id": "Int64",
                "author_id": "string",
                "inbound": "boolean",
                "response_tweet_id": "string",
                "in_response_to_tweet_id": "string",
                "text": "string",
                "created_at": "string",
            },
        ),
        start=1,
    ):
        total_rows += len(chunk)

        brand_mask = (
            chunk["author_id"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq(BRAND.lower())
        )

        brand_chunk = chunk.loc[brand_mask]

        brand_rows += len(brand_chunk)

        for tweet_id in brand_chunk["tweet_id"].dropna():
            brand_tweet_ids.add(str(tweet_id))

        for parent_id in brand_chunk["in_response_to_tweet_id"].dropna():
            parent_tweet_ids.add(str(parent_id))

        print(
            f"Processed chunk {chunk_number:02d} | "
            f"rows={total_rows:,} | "
            f"AmazonHelp={brand_rows:,}"
        )

    print()
    print(f"AmazonHelp tweets found: {len(brand_tweet_ids):,}")
    print(f"Parent tweet IDs needed:  {len(parent_tweet_ids):,}")

    print()
    print("PASS 2: Retrieving customer messages...")

    parent_rows = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            dtype={
                "tweet_id": "Int64",
                "author_id": "string",
                "inbound": "boolean",
                "response_tweet_id": "string",
                "in_response_to_tweet_id": "string",
                "text": "string",
                "created_at": "string",
            },
        ),
        start=1,
    ):
        chunk_ids = chunk["tweet_id"].astype("string")

        mask = chunk_ids.isin(parent_tweet_ids)

        matched = chunk.loc[mask]

        for _, row in matched.iterrows():
            tweet_id = str(row["tweet_id"])

            parent_rows[tweet_id] = {
                "tweet_id": tweet_id,
                "author_id": str(row["author_id"]),
                "inbound": row["inbound"],
                "created_at": row["created_at"],
                "text": str(row["text"]) if pd.notna(row["text"]) else "",
            }

        print(
            f"Processed chunk {chunk_number:02d} | "
            f"customer messages found={len(parent_rows):,}"
        )


    print()
    print("PASS 3: Building customer/support pairs...")

    pairs = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            dtype={
                "tweet_id": "Int64",
                "author_id": "string",
                "inbound": "boolean",
                "response_tweet_id": "string",
                "in_response_to_tweet_id": "string",
                "text": "string",
                "created_at": "string",
            },
        ),
        start=1,
    ):
        brand_mask = (
            chunk["author_id"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq(BRAND.lower())
        )

        brand_chunk = chunk.loc[brand_mask]

        for _, brand_row in brand_chunk.iterrows():

            parent_id = brand_row["in_response_to_tweet_id"]

            if pd.isna(parent_id):
                continue

            parent_id = str(parent_id)

            customer = parent_rows.get(parent_id)

            if customer is None:
                continue

            customer_text = customer["text"].strip()

            brand_text = (
                str(brand_row["text"])
                if pd.notna(brand_row["text"])
                else ""
            ).strip()

            if not customer_text or not brand_text:
                continue

            pairs.append(
                {
                    "customer_tweet_id": customer["tweet_id"],
                    "brand_tweet_id": str(brand_row["tweet_id"]),
                    "customer_author_id": customer["author_id"],
                    "brand": BRAND,
                    "customer_created_at": customer["created_at"],
                    "brand_created_at": brand_row["created_at"],
                    "customer_text": customer_text,
                    "brand_response": brand_text,
                }
            )

        print(
            f"Processed chunk {chunk_number:02d} | "
            f"pairs={len(pairs):,}"
        )


    pairs_df = pd.DataFrame(pairs)

    if pairs_df.empty:
        raise RuntimeError(
            "No customer -> AmazonHelp conversation pairs were found."
        )

    pairs_df = pairs_df.drop_duplicates(
        subset=["customer_tweet_id", "brand_tweet_id"]
    )


    pairs_df = pairs_df[
        (pairs_df["customer_text"].str.len() > 0)
        & (pairs_df["brand_response"].str.len() > 0)
    ]

    output_path = OUTPUT_DIR / "amazonhelp_conversations.csv"

    pairs_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Pairs saved: {len(pairs_df):,}")
    print(f"Output: {output_path}")

    print()
    print("Columns:")
    for column in pairs_df.columns:
        print(f" - {column}")

    print()
    print("Sample conversations:")
    print("-" * 70)

    for _, row in pairs_df.head(5).iterrows():
        print()
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print("AMAZONHELP:")
        print(row["brand_response"])
        print("-" * 70)


if __name__ == "__main__":
    main()