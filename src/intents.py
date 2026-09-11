"""
Rule-based "silver" intent labeler.

In a real deployment we would NOT have gold intent labels for 3M tweets.
The realistic approach (and what we do here) is:
  1. Write a keyword/regex labeler encoding domain knowledge -> "silver" labels
     for the bulk of the data.
  2. Hand-label a small golden set (see eval/build_golden_set.py) for
     evaluation ONLY — never train on it.
  3. Train the actual classifier (src/classify_baseline.py) on the silver
     labels, then measure it against the hand-labeled golden set.

This keeps the eval honest: the golden set is not leaking into training via
some shared heuristic-label pathway that also produced the training labels
1:1 (some correlation is unavoidable since both are keyword-informed, but the
golden set is reviewed and corrected by a human pass — see build_golden_set.py).
"""
import re
from src.config import INTENTS

_PATTERNS = [
    ("cancellation_request", re.compile(r"\bcancel\b|\bcancelled\b|\bcancelling\b", re.I)),
    ("refund_request", re.compile(r"\brefund\b|\bmoney back\b|\breimburse\b", re.I)),
    ("account_access", re.compile(r"\blocked out\b|\bcan'?t log ?in\b|\bpassword\b|\bhacked\b|\baccount\b.*\b(access|login)\b", re.I)),
    ("billing_issue", re.compile(r"\bcharged\b|\bcharge\b|\bbilled\b|\bbilling\b|\bdouble charg", re.I)),
    ("product_defect", re.compile(r"\bbroken\b|\bdamaged\b|\bdefective\b|\bwrong item\b|\bstopped working\b|\bwrong (?:one|product)\b", re.I)),
    ("delivery_delay", re.compile(r"\bdelay\b|\blate\b|\bhasn'?t arrived\b|\bstill (?:not|hasn't)\b.*\b(arrive|here|shown)\b|\bwhere is my\b|\btracking\b", re.I)),
    ("order_status", re.compile(r"\bstatus\b|\border #?\d+\b.*\bstatus\b|\bcheck(ing)? on\b.*\border\b", re.I)),
    ("general_feedback", re.compile(r"\bthank you\b|\bthanks\b|\bamazing\b|\blove\b|\bdisappoint\b|\bquality\b|\bused to be\b", re.I)),
]


def silver_label(text: str) -> str:
    """Return best-guess intent via first matching pattern; fallback general_feedback."""
    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return intent
    return "general_feedback"


def all_intents():
    return list(INTENTS)
