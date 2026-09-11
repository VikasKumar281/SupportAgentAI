import argparse
import re
import pandas as pd

INTENTS = {
    "order_delivery": [
        "delivery", "deliver", "package", "parcel", "shipping", "shipment",
        "tracking", "arrive", "arrival", "late", "delay", "missing order",
        "not received", "where is my order", "courier", "dispatch"
    ],
    "returns_refunds": [
        "return", "refund", "money back", "reimburse", "returned",
        "replacement", "cancel return", "return label"
    ],
    "payment_billing": [
        "payment", "charged", "charge", "billing", "invoice", "card",
        "credit card", "debit card", "price charged", "transaction",
        "unauthorized purchase", "subscription charge"
    ],
    "account_security": [
        "account", "password", "login", "sign in", "security", "hack",
        "hacked", "fraud", "scam", "suspicious", "phishing", "email",
        "locked", "verification"
    ],
    "subscription_prime": [
        "prime", "membership", "prime membership", "trial", "renewal",
        "cancel membership", "prime subscription"
    ],
    "digital_content": [
        "prime video", "video", "movie", "film", "streaming", "kindle",
        "ebook", "book", "audible", "music", "digital"
    ],
    "device_technical": [
        "kindle", "fire tv", "device", "tablet", "screen", "charger",
        "remote", "app", "application", "error", "not working",
        "doesn't work", "technical", "software", "wifi"
    ],
    "product_order_issue": [
        "product", "item", "seller", "product unavailable", "out of stock",
        "wrong item", "damaged", "defective", "product issue", "order issue",
        "coupon", "discount", "wishlist"
    ],
    "customer_service_complaint": [
        "customer service", "support", "agent", "representative", "call",
        "phone", "contact", "no response", "nobody replied", "complaint",
        "unresolved", "waiting", "poor service"
    ],
    "other_non_actionable": [
        "thank you", "thanks", "great", "awesome", "love amazon",
        "feedback", "suggestion", "congratulations", "hello", "hi"
    ]
}

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def score_text(text, keywords):
    score = 0
    for keyword in keywords:
        if keyword in text:
            score += 1
    return score

def suggest_intent(text):
    normalized = normalize(text)
    scores = {
        intent: score_text(normalized, keywords)
        for intent, keywords in INTENTS.items()
    }

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score == 0:
        return "other_non_actionable", 0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "ambiguous", ranked[0][1]

    return best_intent, best_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/intent_discovery_sample.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/intent_discovery_suggestions.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    results = df["customer_text"].fillna("").apply(suggest_intent)

    df["suggested_intent"] = results.apply(lambda x: x[0])
    df["suggestion_score"] = results.apply(lambda x: x[1])

    df.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output: {args.output}")
    print("\nSuggested intent distribution:")
    print(df["suggested_intent"].value_counts())

if __name__ == "__main__":
    main()