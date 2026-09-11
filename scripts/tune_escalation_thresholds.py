import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from escalation_gate import detect_risk, HIGH_RISK_INTENTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_PATH = os.path.join(BASE_DIR, "data", "processed", "golden_set_reviewed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
RETRIEVAL_MATRIX_PATH = os.path.join(BASE_DIR, "models", "leakage_free_matrix.joblib")
RETRIEVAL_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "leakage_free_vectorizer.joblib")
RETRIEVAL_DOCS_PATH = os.path.join(BASE_DIR, "models", "leakage_free_documents.csv")

golden = pd.read_csv(GOLDEN_PATH)
classifier = load(MODEL_PATH)
vectorizer = load(RETRIEVAL_VECTORIZER_PATH)
matrix = load(RETRIEVAL_MATRIX_PATH)
documents = pd.read_csv(RETRIEVAL_DOCS_PATH)

queries = golden["customer_text"].fillna("").astype(str).tolist()
gold_intents = golden["gold_intent"].astype(str).tolist()
gold_escalations = golden["gold_escalate"].astype(bool).tolist()

intent_probabilities = classifier.predict_proba(queries)
predicted_intents = classifier.classes_[np.argmax(intent_probabilities, axis=1)]
intent_confidences = np.max(intent_probabilities, axis=1)

query_matrix = vectorizer.transform(queries)
similarity_matrix = query_matrix @ matrix.T

retrieval_similarities = similarity_matrix.max(axis=1).toarray().ravel()

risk_flags = []
risk_reasons = []

for message in queries:
    reason = detect_risk(message)
    risk_flags.append(reason is not None)
    risk_reasons.append(reason)

results = []

for confidence_threshold in np.arange(0.15, 0.61, 0.05):
    for retrieval_threshold in np.arange(0.10, 0.61, 0.05):
        predictions = []

        for i in range(len(golden)):
            if risk_flags[i]:
                predictions.append(True)
            elif predicted_intents[i] in HIGH_RISK_INTENTS:
                predictions.append(True)
            elif intent_confidences[i] < confidence_threshold:
                predictions.append(True)
            elif retrieval_similarities[i] < retrieval_threshold:
                predictions.append(True)
            else:
                predictions.append(False)

        precision = precision_score(gold_escalations, predictions, zero_division=0)
        recall = recall_score(gold_escalations, predictions, zero_division=0)
        f1 = f1_score(gold_escalations, predictions, zero_division=0)

        results.append({
            "intent_confidence_threshold": round(confidence_threshold, 2),
            "retrieval_similarity_threshold": round(retrieval_threshold, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "escalation_rate": round(np.mean(predictions), 4)
        })

results_df = pd.DataFrame(results)

best_f1 = results_df.sort_values(
    ["f1", "precision"],
    ascending=[False, False]
).head(10)

best_balanced = results_df[
    (results_df["recall"] >= 0.85) &
    (results_df["precision"] >= 0.40)
].sort_values(
    ["f1", "precision"],
    ascending=[False, False]
).head(10)

output_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "escalation_threshold_results.csv"
)

results_df.to_csv(output_path, index=False)

print()
print("Escalation Threshold Tuning")
print("===========================")
print()
print("Examples evaluated:", len(golden))
print()
print("Top 10 by F1:")
print(best_f1.to_string(index=False))
print()
print("Best balanced candidates:")
print(best_balanced.to_string(index=False))
print()
print("Saved to:", output_path)