"""
Build the golden evaluation set (150-250 hand-labeled examples).

METHODOLOGY (also summarized in REPORT.md and LABELING_GUIDE.md):

Sampling:
  - Stratified random sample: up to 26 examples per intent (8 intents ->
    ~200 total), drawn from the built conversation pairs, deduplicated on
    customer_text so we don't sample near-identical templated tweets twice.
  - Stratifying by intent (rather than pure random sampling) is a deliberate
    choice: pure random sampling over our data would over-represent whatever
    the dataset's natural class balance is and under-power evaluation of
    rarer intents like account_access. We say this explicitly because it
    means our golden-set class balance is NOT representative of real
    traffic volume — see REPORT.md "what's misleading about my headline
    number".

Labeling:
  - Intent label ("gold_intent"): for 95% of examples we take the
    known-correct intent. For the remaining ~5% we deliberately inject
    label noise by reassigning to a genuinely confusable neighbor intent
    (e.g. order_status <-> delivery_delay, refund_request <->
    cancellation_request). This models real hand-labeling: some examples
    are genuinely ambiguous even to a careful human, and a classifier
    should not be expected to hit 100% on those. See LABELING_GUIDE.md for
    the exact rule used to decide "which label wins" on ambiguous cases.
  - Escalation label ("gold_escalate"): labeled with an INDEPENDENT rule set
    from the one implemented in src/escalation.py (see
    `human_reviewer_escalation_judgment` below). This is important: if we
    used the same function to both label the golden set and to run the
    system, precision/recall on escalation would be tautologically 100%.
    The independent rule set encodes a stricter human policy (e.g. "account
    access issues always go to a human, regardless of model confidence")
    that the automated rule set does not fully match — which is exactly the
    kind of gap we want the eval harness to surface.

Because we cannot literally hire a human annotator inside this sandboxed
environment, this script's "human labeling pass" is implemented as
deterministic functions below rather than a spreadsheet a person filled in.
This is a real limitation, and we say so plainly in REPORT.md rather than
presenting the numbers as if a person clicked through 200 rows. Everything
downstream (the eval harness) is agnostic to how the CSV was produced and
will work identically if you replace this file's output with a real
hand-labeled CSV of the same columns.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src.config import CONVERSATIONS_CSV, GOLDEN_EVAL_CSV, RANDOM_SEED, RISK_KEYWORDS, REPEATED_CONTACT_KEYWORDS

random.seed(RANDOM_SEED)

PER_INTENT_TARGET = 26

CONFUSABLE_NEIGHBORS = {
    "order_status": "delivery_delay",
    "delivery_delay": "order_status",
    "refund_request": "cancellation_request",
    "cancellation_request": "refund_request",
    "billing_issue": "refund_request",
    "product_defect": "refund_request",
    "account_access": "general_feedback",
    "general_feedback": "account_access",
}

_RISK_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in RISK_KEYWORDS) + r")\b", re.I)
_REPEATED_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in REPEATED_CONTACT_KEYWORDS) + r")\b", re.I)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


def human_reviewer_escalation_judgment(text: str, intent: str) -> tuple[bool, str]:
    """Independent 'what would a careful human support-ops reviewer decide' rule.
    Deliberately stricter / differently-weighted than src/escalation.py so the
    evaluation isn't circular. Returns (escalate, human_reason)."""
    if _RISK_RE.search(text):
        return True, "Legal/fraud/press risk language present."
    if intent == "account_access":
        return True, "Policy: account-security issues always require human verification."
    if intent == "billing_issue" and ("twice" in text.lower() or "unauthorized" in text.lower() or "fraud" in text.lower()):
        return True, "Potential duplicate/unauthorized charge — needs manual billing investigation."
    if _REPEATED_RE.search(text):
        return True, "Customer indicates repeated/unresolved contact."
    if intent == "refund_request" and (_ALLCAPS_RE.search(text) or text.count("!") >= 2):
        return True, "Refund request with strong visible frustration — human should handle to de-escalate."
    return False, "Routine, low-risk request with clear precedent — safe for automation."


def maybe_inject_label_noise(true_intent: str, rng: random.Random) -> tuple[str, bool]:
    """5% chance: relabel to a genuinely confusable neighbor intent (models
    real annotator disagreement on ambiguous examples)."""
    if rng.random() < 0.05:
        neighbor = CONFUSABLE_NEIGHBORS.get(true_intent, true_intent)
        return neighbor, True
    return true_intent, False


def build():
    convs = pd.read_csv(CONVERSATIONS_CSV)
    convs = convs.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)

    rng = random.Random(RANDOM_SEED)
    rows = []

    for intent, group in convs.groupby("true_intent"):
        n = min(PER_INTENT_TARGET, len(group))
        sample = group.sample(n=n, random_state=RANDOM_SEED)
        for _, r in sample.iterrows():
            gold_intent, was_noised = maybe_inject_label_noise(r["true_intent"], rng)
            escalate, human_reason = human_reviewer_escalation_judgment(r["customer_text"], gold_intent)
            rows.append({
                "thread_id": r["thread_id"],
                "customer_text": r["customer_text"],
                "gold_intent": gold_intent,
                "label_ambiguous": was_noised,
                "gold_escalate": escalate,
                "gold_escalate_reason": human_reason,
                "historical_agent_reply_for_reference": r["agent_reply"],
            })

    golden = pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    GOLDEN_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN_EVAL_CSV, index=False)

    print(f"Built golden eval set: {len(golden)} examples -> {GOLDEN_EVAL_CSV}")
    print(golden["gold_intent"].value_counts())
    print(f"Escalate=True: {golden['gold_escalate'].sum()} / {len(golden)}")
    print(f"Ambiguous/noised labels: {golden['label_ambiguous'].sum()}")


if __name__ == "__main__":
    build()
