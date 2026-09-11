"""
Basic sanity tests. Run with: pytest tests/ -v
These are NOT the evaluation harness (that's eval/run_eval.py) — they just
guard against the pipeline crashing or returning malformed output.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.config import CONVERSATIONS_CSV, MODEL_PATH
from src.escalation import decide_escalation
from src.intents import silver_label
from src.retrieval import HistoricalResolutionRetriever


@pytest.fixture(scope="module")
def conversations():
    if not CONVERSATIONS_CSV.exists():
        pytest.skip("Run scripts/run_all.sh (or data prep steps) before tests.")
    return pd.read_csv(CONVERSATIONS_CSV)


def test_silver_label_returns_valid_intent():
    from src.config import INTENTS
    assert silver_label("please refund my order") in INTENTS
    assert silver_label("where is my package") in INTENTS


def test_escalation_risk_keyword_always_escalates():
    result = decide_escalation("I'm calling my lawyer about this fraud", "billing_issue", 0.99, 0.9)
    assert result["escalate"] is True
    assert result["rule"] == "risk_keyword"


def test_escalation_low_confidence_escalates():
    result = decide_escalation("hello", "general_feedback", 0.2, 0.9)
    assert result["escalate"] is True
    assert result["rule"] == "low_confidence_classification"


def test_escalation_confident_grounded_low_risk_auto_handles():
    result = decide_escalation("what's the status of my order", "order_status", 0.9, 0.9)
    assert result["escalate"] is False


def test_retrieval_returns_k_results(conversations):
    retriever = HistoricalResolutionRetriever(conversations)
    results = retriever.top_k("where is my package, it's late", k=3)
    assert len(results) <= 3
    assert all("similarity" in r for r in results)


def test_pipeline_end_to_end(conversations):
    if not MODEL_PATH.exists():
        pytest.skip("Train the baseline classifier first (src/classify_baseline.py).")
    from src.pipeline import AgentPipeline
    pipeline = AgentPipeline(conversations_df=conversations)
    result = pipeline.handle("where is my order #123456, it's 5 days late")
    assert "predicted_intent" in result
    assert "draft_reply" in result
    assert isinstance(result["escalate"], (bool,))
