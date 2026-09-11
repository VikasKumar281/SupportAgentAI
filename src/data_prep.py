"""
Turn the raw Kaggle-schema tweet table into (customer_text, agent_reply)
pairs for one brand. Works identically whether data/raw_tweets.csv is our
synthetic generator's output or the real twcs.csv from Kaggle (same columns).
"""
import re
import pandas as pd
from src.config import BRAND, RAW_TWEETS_CSV, CONVERSATIONS_CSV


def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", "", text)          # strip @mentions (incl. the brand handle)
    text = re.sub(r"http\S+", "", text)         # strip links
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw(path=RAW_TWEETS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    return df


def build_conversations(df: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """
    For every brand outbound reply, find the inbound tweet it responded to.
    Returns one row per (customer_msg -> brand_reply) pair, which is the
    atomic unit our classifier/reply-generator/escalation logic operates on.
    """
    by_id = df.set_index("tweet_id", drop=False)
    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)]

    records = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "":
            continue
        if parent_id not in by_id.index:
            continue
        cust_row = by_id.loc[parent_id]
        if isinstance(cust_row, pd.DataFrame):  # duplicate ids, defensive
            cust_row = cust_row.iloc[0]
        if not cust_row["inbound"]:
            continue

        records.append({
            "thread_id": cust_row["tweet_id"],
            "customer_tweet_id": cust_row["tweet_id"],
            "agent_tweet_id": reply_row["tweet_id"],
            "customer_text_raw": cust_row["text"],
            "customer_text": clean_text(cust_row["text"]),
            "agent_reply_raw": reply_row["text"],
            "agent_reply": clean_text(reply_row["text"]),
            "true_intent": cust_row.get("_true_intent", None),  # only present for synthetic data
        })

    out = pd.DataFrame(records)
    out = out.dropna(subset=["customer_text", "agent_reply"])
    out = out[out["customer_text"].str.len() > 3]
    return out.reset_index(drop=True)


def main():
    raw = load_raw()
    convs = build_conversations(raw)
    CONVERSATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    convs.to_csv(CONVERSATIONS_CSV, index=False)
    print(f"Built {len(convs)} conversation pairs for brand={BRAND} -> {CONVERSATIONS_CSV}")


if __name__ == "__main__":
    main()
