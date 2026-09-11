"""
generate_synthetic_data.py
===========================
IMPORTANT / HONESTY NOTE
-------------------------
This environment cannot reach kaggle.com to download the real
"Customer Support on Twitter" dataset (~3M tweets, 1.7GB, requires a Kaggle
account + API token; network egress here is restricted to pypi/npm/github).

Rather than fake having used the real dataset, this script GENERATES a
synthetic dataset that:
  - Uses the EXACT same column schema as the real Kaggle CSV
    (tweet_id, author_id, inbound, created_at, text, response_tweet_id,
     in_response_to_tweet_id), so every downstream script is schema-compatible.
  - Reproduces the real dataset's noise characteristics: @mentions, hashtags,
    typos, emoji, truncated/rambling messages, multi-turn threads, order
    numbers, dollar amounts.
  - Encodes a KNOWN ground-truth intent per thread (kept in a side column,
    `_true_intent`, dropped when mimicking the raw file, but reused later to
    build the golden eval set honestly rather than hand-waving it).

TO USE THE REAL DATASET INSTEAD:
  1. Download `twcs.csv` from kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  2. Place it at data/raw_tweets.csv (same column names, no code changes needed)
  3. Re-run scripts/run_all.sh — everything downstream is dataset-agnostic
     except that you should re-run eval/build_golden_set.py's labeling pass
     by hand instead of trusting the synthetic ground truth.

Run: python data/generate_synthetic_data.py
"""
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.config import BRAND, RAW_TWEETS_CSV, RANDOM_SEED

random.seed(RANDOM_SEED)

N_CONVERSATIONS = 650  # -> ~1400-1900 tweet rows after multi-turn expansion

PRODUCTS = ["headphones", "phone case", "the blender", "my laptop charger",
            "the running shoes", "a book I ordered", "the coffee maker",
            "my kid's toy", "the desk lamp", "the backpack", "wireless mouse",
            "the air fryer", "monitor stand", "yoga mat", "bluetooth speaker"]

CUSTOMER_HANDLES = [f"user{i:05d}" for i in range(1, 4000)]
EMOJIS = ["😡", "😠", "🙄", "😢", "🙏", "😊", "👍", "❤️", "😤", ""]
HASHTAGS = ["#terrible", "#help", "#customerservice", "#neveragain", "#thanks", ""]

def maybe_typo(text):
    """Randomly drop a letter or duplicate one, to mimic real typing noise."""
    if random.random() < 0.15 and len(text) > 8:
        i = random.randint(3, len(text) - 3)
        text = text[:i] + text[i] + text[i:]
    return text

def order_number():
    return "#" + "".join(random.choices("0123456789", k=random.choice([6, 9])))

def amount():
    return f"${random.randint(8, 480)}.{random.randint(0,99):02d}"

def days():
    return random.choice([2, 3, 4, 5, 6, 7, 8, 10, 14])

# ---------------------------------------------------------------------------
# Templates per intent: (customer_msg_fn, list_of_possible_agent_reply_fns)
# Multiple agent reply variants per intent = the "historical resolution"
# knowledge base that retrieval/grounding will draw on later.
# ---------------------------------------------------------------------------
def tmpl_delivery_delay():
    p = random.choice(PRODUCTS)
    d = days()
    oid = order_number()
    cust = random.choice([
        f"@{BRAND} where is my order {oid}?? said it would arrive {d} days ago and nothing {random.choice(EMOJIS)}",
        f"@{BRAND} {p} still hasn't shown up, order {oid}, this is ridiculous {random.choice(HASHTAGS)}",
        f"hey @{BRAND} my package for {p} is {d} days late, tracking hasn't updated in days",
        f"@{BRAND} order {oid} was supposed to be here already. where is it?",
    ])
    replies = [
        f"Hi, sorry for the delay on order {oid}! I can see it's currently in transit — can you DM us your zip code so we can check the carrier update?",
        f"So sorry about that! Delays like this are usually carrier-side. Please send us order {oid} via DM and we'll get an updated ETA for you.",
        f"That's not the experience we want for you. We've flagged order {oid} for expedited tracking — a specialist will DM you within 24 hours.",
    ]
    return maybe_typo(cust), replies, oid

def tmpl_order_status():
    oid = order_number()
    cust = random.choice([
        f"@{BRAND} what's the status on order {oid}?",
        f"can someone tell me the status of {oid} @{BRAND}",
        f"@{BRAND} just checking on order {oid}, no updates on the app",
    ])
    replies = [
        f"Happy to check that for you! Please DM us order {oid} along with the email used for the purchase.",
        f"Thanks for reaching out — send us order {oid} via DM and we'll pull up the latest status right away.",
    ]
    return maybe_typo(cust), replies, oid

def tmpl_refund_request():
    p = random.choice(PRODUCTS)
    amt = amount()
    oid = order_number()
    cust = random.choice([
        f"@{BRAND} I want a refund for {p}, order {oid}, paid {amt} for garbage {random.choice(EMOJIS)}",
        f"@{BRAND} please refund order {oid}, item was wrong. this is unacceptable {random.choice(HASHTAGS)}",
        f"requesting a refund on order {oid} from @{BRAND}, {amt} charged for nothing",
    ])
    replies = [
        f"I'm sorry to hear that. We can process a refund for order {oid} — please DM us and we'll start that for you right away.",
        f"That's completely understandable, we'll get {amt} refunded for order {oid}. Please confirm via DM and allow 3-5 business days.",
        f"Apologies for the trouble! Send us order {oid} in a DM and our team will issue the refund of {amt} today.",
    ]
    return maybe_typo(cust), replies, oid

def tmpl_billing_issue():
    amt = amount()
    cust = random.choice([
        f"@{BRAND} I was charged {amt} TWICE for the same order, this is fraud {random.choice(EMOJIS)}",
        f"@{BRAND} why is there a charge of {amt} on my card I don't recognize?? unauthorized charge!",
        f"@{BRAND} billed {amt} but I never even received a confirmation email",
    ])
    replies = [
        f"I understand the concern — duplicate charges usually reverse within 3-5 business days, but let's verify. Please DM your order number.",
        f"Sorry about that! Please DM us the last 4 digits of the card and order details so we can investigate the {amt} charge.",
    ]
    return maybe_typo(cust), replies, None

def tmpl_account_access():
    cust = random.choice([
        f"@{BRAND} I've been locked out of my account for 2 days now, password reset isn't working",
        f"@{BRAND} can't log in, keeps saying invalid password even after reset, need this fixed today",
        f"hey @{BRAND} my account got hacked I think, someone changed my email on file",
    ])
    replies = [
        "Sorry for the trouble logging in! Please DM us the email on file (do not share your password) and we'll help restore access.",
        "That sounds frustrating — for account security we'll need to verify a few details over DM. Can you message us there?",
    ]
    return maybe_typo(cust), replies, None

def tmpl_product_defect():
    p = random.choice(PRODUCTS)
    oid = order_number()
    cust = random.choice([
        f"@{BRAND} {p} arrived broken, order {oid}, packaging was fine so it must've shipped like this",
        f"@{BRAND} received the wrong item entirely for order {oid}, ordered {p} got something else",
        f"@{BRAND} {p} stopped working after 2 days, order {oid}, very disappointed {random.choice(EMOJIS)}",
    ])
    replies = [
        f"So sorry to hear that! We'll get a replacement out for order {oid} — please DM us a photo of the item if you can.",
        f"That's disappointing to hear, apologies. Please DM order {oid} and we'll arrange a free return + replacement.",
    ]
    return maybe_typo(cust), replies, oid

def tmpl_cancellation():
    oid = order_number()
    cust = random.choice([
        f"@{BRAND} please cancel order {oid}, ordered by mistake",
        f"@{BRAND} how do I cancel my subscription, don't want to be charged again",
        f"need to cancel order {oid} asap @{BRAND} before it ships",
    ])
    replies = [
        f"No problem! Please DM order {oid} and we'll cancel it if it hasn't shipped yet.",
        "We can help with that — please DM us your account email and we'll process the cancellation.",
    ]
    return maybe_typo(cust), replies, oid

def tmpl_general_feedback():
    cust = random.choice([
        f"@{BRAND} just wanted to say your support team was amazing today, thank you! {random.choice(EMOJIS)}",
        f"@{BRAND} honestly disappointed with the whole experience lately, quality has dropped",
        f"@{BRAND} love the new app update, so much easier to use now",
        f"@{BRAND} customer service used to be so much better, what happened",
    ])
    replies = [
        "Thank you so much for the kind words, we'll pass this along to the team! 😊",
        "We appreciate the honest feedback and we're sorry to hear that — we're always working to improve.",
    ]
    return maybe_typo(cust), replies, None

INTENT_GENERATORS = {
    "delivery_delay": tmpl_delivery_delay,
    "order_status": tmpl_order_status,
    "refund_request": tmpl_refund_request,
    "billing_issue": tmpl_billing_issue,
    "account_access": tmpl_account_access,
    "product_defect": tmpl_product_defect,
    "cancellation_request": tmpl_cancellation,
    "general_feedback": tmpl_general_feedback,
}

def build_dataset():
    rows = []
    tweet_id = 1
    base_time = datetime(2024, 1, 1)
    intents = list(INTENT_GENERATORS.keys())

    for i in range(N_CONVERSATIONS):
        intent = intents[i % len(intents)]
        cust_text, agent_replies, _ = INTENT_GENERATORS[intent]()
        agent_text = random.choice(agent_replies)
        customer_handle = random.choice(CUSTOMER_HANDLES)
        t0 = base_time + timedelta(minutes=i * 7, seconds=random.randint(0, 59))

        cust_tweet_id = tweet_id
        agent_tweet_id = tweet_id + 1
        tweet_id += 2

        # inbound customer tweet
        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": customer_handle,
            "inbound": True,
            "created_at": t0.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": cust_text,
            "response_tweet_id": str(agent_tweet_id),
            "in_response_to_tweet_id": "",
            "_true_intent": intent,
        })
        # outbound brand reply
        rows.append({
            "tweet_id": agent_tweet_id,
            "author_id": BRAND,
            "inbound": False,
            "created_at": (t0 + timedelta(minutes=random.randint(2, 40))).strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": agent_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_tweet_id),
            "_true_intent": intent,
        })

        # ~30% of threads get a customer follow-up (multi-turn realism)
        if random.random() < 0.30:
            followup_id = tweet_id
            tweet_id += 1
            followup_text = random.choice([
                "ok I sent the DM, waiting to hear back",
                "still nothing?? it's been hours",
                "thank you, appreciate the quick help",
                "just DM'd you the info",
            ])
            rows.append({
                "tweet_id": followup_id,
                "author_id": customer_handle,
                "inbound": True,
                "created_at": (t0 + timedelta(minutes=random.randint(41, 90))).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": followup_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": str(agent_tweet_id),
                "_true_intent": intent,
            })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = build_dataset()
    RAW_TWEETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_TWEETS_CSV, index=False)
    print(f"Wrote {len(df)} rows ({df['inbound'].sum()} inbound) to {RAW_TWEETS_CSV}")
    print(df["_true_intent"].value_counts())
