import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from escalation_gate import decide_escalation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_PATH = os.path.join(BASE_DIR, "data", "processed", "golden_set_reviewed.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
VECTOR_PATH = os.path.join(BASE_DIR, "models", "leakage_free_vectorizer.joblib")
MATRIX_PATH = os.path.join(BASE_DIR, "models", "leakage_free_matrix.joblib")
DATA_PATH = os.path.join(BASE_DIR, "models", "leakage_free_documents.csv")

golden = pd.read_csv(GOLDEN_PATH)
classifier = joblib.load(CLASSIFIER_PATH)
vectorizer = joblib.load(VECTOR_PATH)
matrix = joblib.load(MATRIX_PATH)
documents = pd.read_csv(DATA_PATH, low_memory=False)

def clean_response(text):
    text = str(text)
    text = re.sub(r"@\d+", "", text)
    text = re.sub(r"https?://t\.co/\S+", "", text)
    text = re.sub(r"\^?[A-Z]{2}\s*$", "", text)
    text = re.sub(r"\s*(?:You can check here|Check here|See here):?\s*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.!?,])", r"\1", text)
    return text

messages = golden["customer_text"].fillna("").astype(str).tolist()

probabilities = classifier.predict_proba(messages)
predicted_intents = classifier.classes_[np.argmax(probabilities, axis=1)]
intent_confidences = np.max(probabilities, axis=1)

query_matrix = vectorizer.transform(messages)
similarity_matrix = query_matrix @ matrix.T

retrieval_indices = np.argmax(similarity_matrix.toarray(), axis=1)
retrieval_similarities = similarity_matrix.max(axis=1).toarray().ravel()

predicted_escalations = []
escalation_reasons = []
draft_replies = []
retrieved_customer_texts = []
retrieved_responses = []

for i, index in enumerate(retrieval_indices):
    retrieved_customer = str(documents.iloc[index]["customer_text"])
    retrieved_response = clean_response(documents.iloc[index]["brand_response"])

    escalate, reason = decide_escalation(
        messages[i],
        predicted_intents[i],
        float(intent_confidences[i]),
        float(retrieval_similarities[i])
    )

    predicted_escalations.append(escalate)
    escalation_reasons.append(reason)
    draft_replies.append(retrieved_response)
    retrieved_customer_texts.append(retrieved_customer)
    retrieved_responses.append(retrieved_response)

gold_intents = golden["gold_intent"].astype(str)
gold_escalations = golden["gold_escalate"].astype(bool)

intent_accuracy = accuracy_score(gold_intents, predicted_intents)
intent_macro_f1 = f1_score(gold_intents, predicted_intents, average="macro", zero_division=0)

escalation_precision = precision_score(
    gold_escalations,
    predicted_escalations,
    zero_division=0
)

escalation_recall = recall_score(
    gold_escalations,
    predicted_escalations,
    zero_division=0
)

escalation_f1 = f1_score(
    gold_escalations,
    predicted_escalations,
    zero_division=0
)

auto_handle_rate = 1 - np.mean(predicted_escalations)

results = golden.copy()

results["predicted_intent"] = predicted_intents
results["intent_confidence"] = intent_confidences
results["retrieved_customer_text"] = retrieved_customer_texts
results["retrieved_brand_response"] = retrieved_responses
results["retrieval_similarity"] = retrieval_similarities
results["predicted_escalate"] = predicted_escalations
results["escalation_reason"] = escalation_reasons
results["draft_reply"] = draft_replies
results["intent_correct"] = results["gold_intent"] == results["predicted_intent"]
results["escalation_correct"] = results["gold_escalate"] == results["predicted_escalate"]

output_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "agent_evaluation_200.csv"
)

results.to_csv(output_path, index=False)

print()
print("Agent Evaluation")
print("================")
print()
print(f"Examples evaluated: {len(results)}")
print()
print(f"Intent Accuracy: {intent_accuracy:.4f}")
print(f"Intent Macro F1: {intent_macro_f1:.4f}")
print()
print(f"Escalation Precision: {escalation_precision:.4f}")
print(f"Escalation Recall: {escalation_recall:.4f}")
print(f"Escalation F1: {escalation_f1:.4f}")
print(f"Auto-Handle Rate: {auto_handle_rate:.4f}")
print()
print("Escalation Reasons:")
print(results["escalation_reason"].value_counts().to_string())
print()
print("Intent Results:")
print(
    pd.crosstab(
        results["gold_intent"],
        results["predicted_intent"]
    ).to_string()
)
print()
print("Saved to:", output_path)