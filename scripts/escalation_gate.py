import re

HIGH_RISK_INTENTS = {
    "account_security",
    "payment_billing"
}

LOW_CONFIDENCE_THRESHOLD = 0.20
LOW_RETRIEVAL_THRESHOLD = 0.20

def detect_risk(message):
    text = message.lower()

    patterns = {
        "security_or_account_risk": [
            "hacked",
            "hack",
            "stolen account",
            "account compromised",
            "someone accessed",
            "unauthorized access",
            "fraud",
            "identity theft"
        ],
        "payment_or_financial_risk": [
            "charged twice",
            "charged me twice",
            "unauthorized charge",
            "unknown charge",
            "fraudulent charge",
            "credit card fraud"
        ],
        "legal_or_high_risk": [
            "lawyer",
            "legal action",
            "lawsuit",
            "sue",
            "court",
            "attorney"
        ],
        "explicit_human_request": [
            "human agent",
            "real person",
            "speak to an agent",
            "talk to an agent",
            "customer service representative",
            "supervisor"
        ]
    }

    for reason, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text:
                return reason

    return None

def decide_escalation(message, intent, intent_confidence, retrieval_similarity):
    risk_reason = detect_risk(message)

    if risk_reason:
        return True, risk_reason

    if intent in HIGH_RISK_INTENTS:
        return True, "high_risk_intent"

    if intent_confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, "low_intent_confidence"

    if retrieval_similarity < LOW_RETRIEVAL_THRESHOLD:
        return True, "low_historical_similarity"

    return False, "safe_to_auto_handle"