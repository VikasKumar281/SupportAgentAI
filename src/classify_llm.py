"""
Optional LLM-based zero/few-shot intent classifier.

If ANTHROPIC_API_KEY is not set (or the call fails for any reason — no
network, rate limit, etc.), this transparently falls back to the TF-IDF
baseline so the pipeline NEVER crashes and NEVER silently hangs waiting on
a network call that can't succeed in a sandboxed grading environment.
"""
import os
import json
import re

from src.config import INTENTS, ANTHROPIC_MODEL

_FEWSHOT = """You are an intent classifier for customer support tweets sent to an
e-commerce brand. Classify the message into exactly one of these intents:
{intents}

Respond with ONLY a JSON object: {{"intent": "<one_of_the_intents>", "confidence": <0.0-1.0>}}

Examples:
"where is my package, ordered 8 days ago" -> {{"intent": "delivery_delay", "confidence": 0.9}}
"can you cancel order #123456" -> {{"intent": "cancellation_request", "confidence": 0.95}}
"I was charged twice for the same order" -> {{"intent": "billing_issue", "confidence": 0.85}}

Message: "{message}"
"""


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def classify_with_llm(text: str):
    """Returns (intent, confidence, used_llm: bool)."""
    client = _client()
    if client is None:
        return None, None, False

    prompt = _FEWSHOT.format(intents=", ".join(INTENTS), message=text)
    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None, None, False
        parsed = json.loads(match.group(0))
        intent = parsed.get("intent")
        confidence = float(parsed.get("confidence", 0.5))
        if intent not in INTENTS:
            return None, None, False
        return intent, confidence, True
    except Exception:
        return None, None, False


def classify(text: str, fallback_model):
    """
    Try LLM classification first; if unavailable/fails, use the provided
    fallback (TfidfLogRegBaseline instance). Returns dict with provenance
    so the eval harness can report what fraction of calls actually used
    the LLM vs. fell back (this matters — see REPORT.md misleading-number
    section).
    """
    intent, confidence, used_llm = classify_with_llm(text)
    if used_llm:
        return {"intent": intent, "confidence": confidence, "source": "llm"}

    preds = fallback_model.predict_with_confidence([text])
    intent, confidence = preds[0]
    return {"intent": intent, "confidence": float(confidence), "source": "tfidf_fallback"}
