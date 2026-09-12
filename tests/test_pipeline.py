import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pandas as pd

from escalation_gate import decide_escalation, detect_risk


ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = ROOT / "data" / "processed" / "amazonhelp_conversations.csv"
GOLDEN = ROOT / "data" / "processed" / "golden_set_reviewed.csv"
MODEL = ROOT / "models" / "intent_classifier.joblib"


def test_conversation_dataset_exists():
    assert CONVERSATIONS.exists()
    df = pd.read_csv(CONVERSATIONS, nrows=5)
    required = {
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response",
    }
    assert required.issubset(df.columns)


def test_golden_set_exists():
    assert GOLDEN.exists()
    df = pd.read_csv(GOLDEN)
    assert len(df) == 200
    assert "gold_intent" in df.columns
    assert "gold_escalate" in df.columns


def test_classifier_model_exists():
    assert MODEL.exists()


def test_security_risk_escalates():
    escalate, reason = decide_escalation(
        "My account was hacked",
        "account_security",
        0.95,
        0.80,
    )
    assert escalate is True
    assert reason == "security_or_account_risk"


def test_payment_risk_escalates():
    escalate, reason = decide_escalation(
        "I see an unauthorized charge on my card",
        "payment_billing",
        0.95,
        0.80,
    )
    assert escalate is True
    assert reason == "payment_or_financial_risk"


def test_explicit_human_request_escalates():
    escalate, reason = decide_escalation(
        "I want to speak to a human agent",
        "customer_service_complaint",
        0.95,
        0.80,
    )
    assert escalate is True
    assert reason == "explicit_human_request"


def test_low_confidence_escalates():
    escalate, reason = decide_escalation(
        "Where is my order?",
        "order_delivery",
        0.10,
        0.80,
    )
    assert escalate is True
    assert reason == "low_intent_confidence"


def test_low_similarity_escalates():
    escalate, reason = decide_escalation(
        "I need help with something unusual",
        "other_non_actionable",
        0.80,
        0.05,
    )
    assert escalate is True
    assert reason == "low_historical_similarity"


def test_confident_grounded_low_risk_auto_handles():
    escalate, reason = decide_escalation(
        "Where is my order?",
        "order_delivery",
        0.90,
        0.80,
    )
    assert escalate is False
    assert reason == "safe_to_auto_handle"


def test_detect_risk_returns_none_for_normal_message():
    assert detect_risk("Where is my package?") is None