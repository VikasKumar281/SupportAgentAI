"""
Escalation decision: auto-handle vs. route to a human, WITH a stated reason.

Deliberately rule-based rather than another LLM call: escalation is a safety
gate, and gates should be auditable/deterministic, not subject to LLM
sampling variance. See REPORT.md "problem framing" for the reasoning.

Rules are checked in priority order; the FIRST one that fires wins (so the
reason reported is always the single most important one, not a list).
"""
import re
from src.config import (
    CONFIDENCE_THRESHOLD, RETRIEVAL_SIM_THRESHOLD, HIGH_RISK_INTENTS,
    RISK_KEYWORDS, REPEATED_CONTACT_KEYWORDS,
)

# NOTE: word boundaries (\b) matter here. An earlier version without them
# matched "again" inside the hashtag "#neveragain", silently escalating
# messages that were never repeated contacts. Caught by inspecting real
# predictions.csv output during development — see REPORT.md failure #1.
_RISK_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in RISK_KEYWORDS) + r")\b", re.I)
_REPEATED_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in REPEATED_CONTACT_KEYWORDS) + r")\b", re.I)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


def decide_escalation(text: str, intent: str, confidence: float, top_similarity: float):
    """
    Returns dict: {escalate: bool, reason: str, rule: str}
    `rule` is a short machine-readable tag; `reason` is the human-readable
    sentence a support lead would see.
    """
    if _RISK_RE.search(text):
        return {"escalate": True, "rule": "risk_keyword",
                "reason": "Message contains legal/fraud/press-risk language (e.g. 'chargeback', 'lawyer', 'fraud') — needs human judgment."}

    if confidence < CONFIDENCE_THRESHOLD:
        return {"escalate": True, "rule": "low_confidence_classification",
                "reason": f"Intent classifier confidence ({confidence:.2f}) is below threshold ({CONFIDENCE_THRESHOLD}) — too uncertain to auto-handle."}

    if top_similarity < RETRIEVAL_SIM_THRESHOLD:
        return {"escalate": True, "rule": "no_grounded_precedent",
                "reason": f"No sufficiently similar historical resolution found (best match similarity {top_similarity:.2f}) — auto-reply would not be grounded."}

    if _REPEATED_RE.search(text):
        return {"escalate": True, "rule": "repeated_unresolved_contact",
                "reason": "Customer indicates this is a repeated/unresolved contact — escalate to avoid compounding frustration with another templated reply."}

    if intent in HIGH_RISK_INTENTS and (_ALLCAPS_RE.search(text) or text.count("!") >= 2):
        return {"escalate": True, "rule": "high_risk_intent_with_high_emotion",
                "reason": f"High-risk intent ('{intent}') combined with signs of strong frustration — safer for a human to handle money/account actions here."}

    return {"escalate": False, "rule": "auto_handle",
            "reason": "Confident classification, grounded historical precedent, no risk signals — safe to auto-handle."}
