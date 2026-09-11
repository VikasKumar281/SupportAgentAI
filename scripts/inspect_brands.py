from collections import Counter
import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
CHUNK_SIZE = 100_000

brand_counts = Counter()
total_rows = 0
inbound_rows = 0
outbound_rows = 0

print("Reading real TWCS dataset in chunks...")
print()

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
        },
    ),
    start=1,
):
    total_rows += len(chunk)

    inbound = chunk["inbound"].eq(True)
    outbound = chunk["inbound"].eq(False)

    inbound_rows += inbound.sum()
    outbound_rows += outbound.sum()

    # Brand/support accounts are represented by outbound tweets.
    outbound_authors = (
        chunk.loc[outbound, "author_id"]
        .dropna()
        .astype(str)
    )

    brand_counts.update(outbound_authors)

    print(
        f"Processed chunk {chunk_number} | "
        f"rows so far: {total_rows:,}"
    )

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total rows:     {total_rows:,}")
print(f"Inbound tweets: {inbound_rows:,}")
print(f"Outbound tweets:{outbound_rows:,}")

print()
print("=" * 70)
print("TOP SUPPORT / BRAND ACCOUNTS")
print("=" * 70)

for rank, (author, count) in enumerate(
    brand_counts.most_common(30),
    start=1,
):
    print(f"{rank:2}. {author:<30} {count:>10,}")