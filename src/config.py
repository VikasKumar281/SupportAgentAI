"""
Central configuration for the Hiver support-agent pipeline.
Change BRAND / paths here to point at a different subset of the dataset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Brand selection ----
# The real Kaggle "Customer Support on Twitter" dataset has dozens of brand
# handles (AmazonHelp, AppleSupport, Delta, SpotifyCares, ...). We picked
# AmazonHelp: high volume, diverse intents (orders/refunds/delivery/billing),
# good for demonstrating grounded retrieval on repeated issue types.
BRAND = "AmazonHelp"

# ---- Data paths ----
RAW_TWEETS_CSV = ROOT / "data" / "raw_tweets.csv"          # Kaggle-schema CSV
CONVERSATIONS_CSV = ROOT / "data" / "conversations.csv"     # built customer<->agent pairs
GOLDEN_EVAL_CSV = ROOT / "data" / "golden_eval_set.csv"
HUMAN_CALIBRATION_CSV = ROOT / "eval" / "human_calibration_labels.csv"

OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUTS_DIR / "baseline_classifier.joblib"

# ---- Intent taxonomy ----
# Deliberately small (8 classes) and mutually-exclusive-ish. Defined by
# reading ~150 raw AmazonHelp threads and clustering by "what does the
# customer actually want". See REPORT.md "Problem framing".
INTENTS = [
    "delivery_delay",       # "where is my package", "still not arrived"
    "order_status",         # generic "what's the status of order #X"
    "refund_request",       # "I want my money back", "refund my order"
    "billing_issue",        # wrong charge, double charge, unexpected charge
    "account_access",       # locked out, can't log in, password reset
    "product_defect",       # item broken / wrong item / damaged
    "cancellation_request", # cancel order / cancel subscription
    "general_feedback",     # compliments, vague complaints, doesn't fit above
]

# ---- Escalation thresholds (see src/escalation.py for full rule set) ----
CONFIDENCE_THRESHOLD = 0.55      # below this, classifier is "not sure enough"
RETRIEVAL_SIM_THRESHOLD = 0.15   # below this, no good grounded precedent exists
HIGH_RISK_INTENTS = {"billing_issue", "account_access", "refund_request"}

RISK_KEYWORDS = [
    "lawyer", "sue", "legal action", "chargeback", "fraud", "hacked",
    "unauthorized", "bbb", "better business bureau", "reporter", "press",
    "class action", "attorney", "fcc", "ftc",
]
REPEATED_CONTACT_KEYWORDS = [
    "again", "third time", "second time", "still not", "still havent",
    "still haven't", "already asked", "asked twice", "no response",
    "ignored", "week ago", "days ago and nothing",
]

RANDOM_SEED = 42

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
