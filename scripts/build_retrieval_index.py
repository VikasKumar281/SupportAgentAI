import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "amazonhelp_conversations.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "models")
VECTOR_PATH = os.path.join(OUTPUT_DIR, "retrieval_vectorizer.joblib")
MATRIX_PATH = os.path.join(OUTPUT_DIR, "retrieval_matrix.joblib")
DATA_PATH = os.path.join(OUTPUT_DIR, "retrieval_documents.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

df["customer_text"] = df["customer_text"].fillna("").astype(str)
df["brand_response"] = df["brand_response"].fillna("").astype(str)

df = df[
    (df["customer_text"].str.strip() != "") &
    (df["brand_response"].str.strip() != "")
].copy()

df = df.drop_duplicates(subset=["customer_text", "brand_response"])

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=200000
)

matrix = vectorizer.fit_transform(df["customer_text"])

joblib.dump(vectorizer, VECTOR_PATH)
joblib.dump(matrix, MATRIX_PATH)

df.to_csv(DATA_PATH, index=False)

print()
print("Retrieval Index")
print("===============")
print()
print(f"Documents: {len(df)}")
print(f"Features: {len(vectorizer.get_feature_names_out())}")
print(f"Matrix Shape: {matrix.shape}")
print()
print(f"Saved vectorizer to: {VECTOR_PATH}")
print(f"Saved matrix to: {MATRIX_PATH}")
print(f"Saved documents to: {DATA_PATH}")