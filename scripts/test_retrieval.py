import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_PATH = os.path.join(BASE_DIR, "models", "retrieval_vectorizer.joblib")
MATRIX_PATH = os.path.join(BASE_DIR, "models", "retrieval_matrix.joblib")
DATA_PATH = os.path.join(BASE_DIR, "models", "retrieval_documents.csv")

vectorizer = joblib.load(VECTOR_PATH)
matrix = joblib.load(MATRIX_PATH)

import pandas as pd

df = pd.read_csv(DATA_PATH)

query = input("Customer message: ").strip()

query_vector = vectorizer.transform([query])

scores = matrix.dot(query_vector.T).toarray().ravel()

top_indices = np.argsort(scores)[::-1][:5]

print()
print("Top Historical Matches")
print("======================")

for rank, index in enumerate(top_indices, start=1):
    row = df.iloc[index]

    print()
    print(f"Match {rank}")
    print(f"Similarity: {scores[index]:.4f}")
    print(f"Customer: {row['customer_text']}")
    print(f"AmazonHelp: {row['brand_response']}")