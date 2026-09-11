import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "golden_set_reviewed.csv"
)

VECTOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "leakage_free_vectorizer.joblib"
)

MATRIX_PATH = os.path.join(
    BASE_DIR,
    "models",
    "leakage_free_matrix.joblib"
)

DOCUMENTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "leakage_free_documents.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "retrieval_leakage_free_results.csv"
)

golden = pd.read_csv(
    GOLDEN_PATH,
    low_memory=False
).fillna("")

vectorizer = joblib.load(VECTOR_PATH)
matrix = joblib.load(MATRIX_PATH)

documents = pd.read_csv(
    DOCUMENTS_PATH,
    low_memory=False
).fillna("")

results = []

for _, row in golden.iterrows():
    message = str(row["customer_text"])

    query_vector = vectorizer.transform([message])
    scores = matrix.dot(query_vector.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:5]

    if len(top_indices) == 0:
        continue

    best_index = top_indices[0]

    results.append({
        "sample_id": row["sample_id"],
        "customer_text": message,
        "retrieved_customer_text": documents.iloc[best_index]["customer_text"],
        "retrieved_brand_response": documents.iloc[best_index]["brand_response"],
        "similarity": float(scores[best_index])
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Proper Leakage-Free Retrieval Evaluation")
print("=========================================")
print()
print(f"Examples evaluated: {len(result_df)}")

if len(result_df):
    print(f"Mean similarity: {result_df['similarity'].mean():.4f}")
    print(f"Median similarity: {result_df['similarity'].median():.4f}")
    print(f"Top similarity: {result_df['similarity'].max():.4f}")
    print(f"Minimum similarity: {result_df['similarity'].min():.4f}")
    print()
    print("Top 10 matches:")
    print(
        result_df
        .sort_values("similarity", ascending=False)
        .head(10)
        .to_string(index=False)
    )

print()
print(f"Saved to: {OUTPUT_PATH}")