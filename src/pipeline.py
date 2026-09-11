"""
The actual "AI support agent": one customer message in, a structured
decision out. This is what eval/run_eval.py drives against the golden set.
"""
import pandas as pd

from src.classify_baseline import TfidfLogRegBaseline
from src.classify_llm import classify as classify_intent
from src.retrieval import HistoricalResolutionRetriever
from src.reply_generator import draft_reply
from src.escalation import decide_escalation
from src.config import CONVERSATIONS_CSV, MODEL_PATH


class AgentPipeline:
    def __init__(self, conversations_df: pd.DataFrame = None, baseline_model=None):
        if conversations_df is None:
            conversations_df = pd.read_csv(CONVERSATIONS_CSV)
        self.conversations_df = conversations_df
        self.retriever = HistoricalResolutionRetriever(conversations_df)

        if baseline_model is None:
            if MODEL_PATH.exists():
                baseline_model = TfidfLogRegBaseline.load()
            else:
                raise FileNotFoundError(
                    f"No trained baseline model at {MODEL_PATH}. Run `python -m src.classify_baseline` first."
                )
        self.baseline_model = baseline_model

    def handle(self, customer_text: str) -> dict:
        cls_result = classify_intent(customer_text, self.baseline_model)
        intent = cls_result["intent"]
        confidence = cls_result["confidence"]

        retrieved = self.retriever.top_k(customer_text, k=3, intent_filter=intent)
        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        reply_result = draft_reply(customer_text, retrieved)

        escalation = decide_escalation(
            text=customer_text, intent=intent, confidence=confidence, top_similarity=top_similarity,
        )

        return {
            "customer_text": customer_text,
            "predicted_intent": intent,
            "intent_confidence": round(confidence, 3),
            "classifier_source": cls_result["source"],
            "top_similarity": round(top_similarity, 3),
            "draft_reply": reply_result["reply"],
            "reply_source": reply_result["source"],
            "grounded_on": reply_result["grounded_on"],
            "escalate": escalation["escalate"],
            "escalation_rule": escalation["rule"],
            "escalation_reason": escalation["reason"],
        }
