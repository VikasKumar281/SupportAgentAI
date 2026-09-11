"""
LLM-as-judge for reply quality, plus agreement-with-human calibration
(assignment requirement #3: "evidence of how well your judge agrees with a
human").

HONESTY NOTE: this sandboxed environment has no human annotator available.
`simulate_human_score()` below is a clearly-labeled SIMULATED proxy for a
human rater — implemented with a deliberately different scoring logic than
the judge (different features, different weights) so agreement isn't
trivially 1.0. It exists to prove the *harness* works end-to-end (a real
deployment would swap this for an actual human-labeled CSV of the same
shape — see eval/human_calibration_labels.csv). We say this again in
REPORT.md's "misleading number" section because reporting judge-vs-human
agreement without this caveat would be dishonest.

Judge rubric (each 1-5, per assignment's reply-quality ask):
  - relevance:     does the reply actually address what the customer asked?
  - groundedness:  is the reply consistent with the retrieved historical
                    precedent (doesn't invent policy/facts)?
  - tone:          appropriately empathetic/professional for the message?
  - actionability: does it give the customer a clear next step?
overall = mean of the four, rounded to nearest 0.5.
"""
import os
import re
import json
import numpy as np

from scipy.stats import pearsonr, spearmanr
from src.config import ANTHROPIC_MODEL

_JUDGE_PROMPT = """You are grading a customer-support reply on Twitter. Score each
dimension from 1 (very poor) to 5 (excellent):
- relevance: does the reply address the customer's actual message?
- groundedness: is it consistent with the retrieved past example(s), without inventing facts (specific amounts, order numbers, promises) not present in the customer message or the examples?
- tone: appropriately empathetic and professional given the customer's tone?
- actionability: does the customer know what to do next?

Customer message: "{customer_text}"
Retrieved historical example(s) used as grounding: {examples}
Agent's draft reply: "{reply}"

Respond with ONLY JSON: {{"relevance": 1-5, "groundedness": 1-5, "tone": 1-5, "actionability": 1-5, "rationale": "one sentence"}}
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


def _heuristic_judge(customer_text: str, reply: str, grounded_examples: list) -> dict:
    """Offline fallback judge used when no API key is configured. Feature-based,
    not a language model — reported separately in results so nobody mistakes
    its scores for genuine LLM judgments (see 'judge_source' field)."""
    cust_words = set(re.findall(r"[a-z]{3,}", customer_text.lower()))
    reply_words = set(re.findall(r"[a-z]{3,}", reply.lower()))
    overlap = len(cust_words & reply_words) / max(1, len(cust_words))
    relevance = 2 + min(3, round(overlap * 6))

    grounded_text = " ".join(grounded_examples).lower()
    reply_numbers = set(re.findall(r"\d{3,}", reply))
    cust_numbers = set(re.findall(r"\d{3,}", customer_text))
    invented_numbers = reply_numbers - cust_numbers - set(re.findall(r"\d{3,}", grounded_text))
    groundedness = 5 if not invented_numbers else 2

    apology_words = {"sorry", "apologize", "understand", "appreciate"}
    has_apology = bool(apology_words & reply_words)
    negative_signals = {"angry", "terrible", "worst", "unacceptable", "ridiculous"} & cust_words
    tone = 5 if (not negative_signals or has_apology) else 2

    actionability = 5 if re.search(r"\bdm\b|\bmessage us\b|\breply\b|\bemail\b", reply.lower()) else 3

    return {"relevance": relevance, "groundedness": groundedness, "tone": tone,
            "actionability": actionability, "rationale": "heuristic feature-based score (no LLM key configured)"}


def score_reply(customer_text: str, reply: str, grounded_examples: list) -> dict:
    client = _client()
    if client is not None:
        prompt = _JUDGE_PROMPT.format(
            customer_text=customer_text,
            examples=json.dumps(grounded_examples[:2]),
            reply=reply,
        )
        try:
            resp = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=200,
                                           messages=[{"role": "user", "content": prompt}])
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                parsed = json.loads(match.group(0))
                dims = [parsed["relevance"], parsed["groundedness"], parsed["tone"], parsed["actionability"]]
                overall = round(float(np.mean(dims)) * 2) / 2
                return {**parsed, "overall": overall, "judge_source": "llm"}
        except Exception:
            pass

    parsed = _heuristic_judge(customer_text, reply, grounded_examples)
    dims = [parsed["relevance"], parsed["groundedness"], parsed["tone"], parsed["actionability"]]
    overall = round(float(np.mean(dims)) * 2) / 2
    return {**parsed, "overall": overall, "judge_source": "heuristic_fallback"}


def simulate_human_score(customer_text: str, reply: str) -> float:
    """
    SIMULATED human rating (see module docstring). Deliberately implemented
    with different signals than the judge above:
      - penalizes replies over 280 chars (a human proofreader would flag
        Twitter-length overflow)
      - rewards replies that mirror a specific noun from the complaint
        (a human notices when a reply feels templated/generic)
      - penalizes overly-terse replies (<20 chars) as unhelpful
    Returns a 1-5 float.
    """
    score = 3.0
    if len(reply) > 280:
        score -= 1.0
    if len(reply) < 20:
        score -= 1.5
    cust_nouns = set(re.findall(r"[a-z]{4,}", customer_text.lower()))
    reply_words = set(re.findall(r"[a-z]{4,}", reply.lower()))
    if cust_nouns & reply_words:
        score += 0.5
    if re.search(r"\bdm\b|\bmessage us\b", reply.lower()):
        score += 0.5
    if re.search(r"sorry|apologize|understand", reply.lower()):
        score += 0.5
    return float(np.clip(round(score * 2) / 2, 1.0, 5.0))


def compute_agreement(judge_scores, human_scores):
    judge_scores = np.array(judge_scores, dtype=float)
    human_scores = np.array(human_scores, dtype=float)

    pearson_r, _ = pearsonr(judge_scores, human_scores) if len(judge_scores) > 1 else (float("nan"), None)
    spearman_r, _ = spearmanr(judge_scores, human_scores) if len(judge_scores) > 1 else (float("nan"), None)
    mae = float(np.mean(np.abs(judge_scores - human_scores)))
    within_1 = float(np.mean(np.abs(judge_scores - human_scores) <= 1.0))

    return {
        "n": len(judge_scores),
        "pearson_r": round(float(pearson_r), 4) if pearson_r == pearson_r else None,
        "spearman_r": round(float(spearman_r), 4) if spearman_r == spearman_r else None,
        "mae": round(mae, 4),
        "within_1_point_agreement_rate": round(within_1, 4),
    }
