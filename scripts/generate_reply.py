import os
import re
import joblib
import numpy as np
import pandas as pd
from escalation_gate import decide_escalation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
VECTOR_PATH = os.path.join(BASE_DIR, "models", "leakage_free_vectorizer.joblib")
MATRIX_PATH = os.path.join(BASE_DIR, "models", "leakage_free_matrix.joblib")
DATA_PATH = os.path.join(BASE_DIR, "models", "leakage_free_documents.csv")

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

def classify_intent(message):
    prediction = classifier.predict([message])[0]
    probabilities = classifier.predict_proba([message])[0]
    confidence = float(np.max(probabilities))
    return prediction, confidence

def retrieve(message, top_k=5):
    query_vector = vectorizer.transform([message])
    scores = matrix.dot(query_vector.T).toarray().ravel()
    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "customer_text": str(documents.iloc[index]["customer_text"]),
            "brand_response": clean_response(documents.iloc[index]["brand_response"]),
            "similarity": float(scores[index])
        })

    return results

def generate_reply(results):
    if not results:
        return "I’m sorry, but I don’t have enough historical information to provide a grounded response."

    best = results[0]

    if best["similarity"] < 0.20:
        return "I’m sorry, but I don’t have enough historical information to provide a grounded response."

    if not best["brand_response"]:
        return "I’m sorry, but I don’t have enough historical information to provide a grounded response."

    return best["brand_response"]

message = input("Customer message: ").strip()

intent, confidence = classify_intent(message)
results = retrieve(message)

retrieval_similarity = results[0]["similarity"] if results else 0.0

escalate, escalation_reason = decide_escalation(
    message,
    intent,
    confidence,
    retrieval_similarity
)

reply = generate_reply(results)

print()
print("Support Agent")
print("=============")
print()
print(f"Intent: {intent}")
print(f"Intent Confidence: {confidence:.4f}")
print(f"Retrieval Similarity: {retrieval_similarity:.4f}")
print()
print(f"Escalate: {escalate}")
print(f"Escalation Reason: {escalation_reason}")
print()

if escalate:
    print("Recommended Action: HUMAN REVIEW")
    print()
    print("Internal Draft Reply:")
else:
    print("Recommended Action: AUTO-HANDLE")
    print()
    print("Reply:")

print(reply)