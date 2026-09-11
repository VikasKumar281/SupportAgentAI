import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONVERSATIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "amazonhelp_conversations.csv"
)

GOLDEN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "golden_set_reviewed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models"
)

VECTOR_PATH = os.path.join(
    OUTPUT_DIR,
    "leakage_free_vectorizer.joblib"
)

MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "leakage_free_matrix.joblib"
)

DOCUMENTS_PATH = os.path.join(
    OUTPUT_DIR,
    "leakage_free_documents.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

conversations = pd.read_csv(
    CONVERSATIONS_PATH,
    low_memory=False
)

golden = pd.read_csv(
    GOLDEN_PATH,
    low_memory=False
)

conversations["customer_text"] = conversations["customer_text"].fillna("").astype(str)
conversations["brand_response"] = conversations["brand_response"].fillna("").astype(str)
golden["customer_text"] = golden["customer_text"].fillna("").astype(str)

golden_texts = set(
    golden["customer_text"]
    .str.strip()
    .tolist()
)

conversations = conversations[
    ~conversations["customer_text"].str.strip().isin(golden_texts)
].copy()

conversations = conversations[
    (conversations["customer_text"].str.strip() != "") &
    (conversations["brand_response"].str.strip() != "")
].copy()

conversations = conversations.drop_duplicates(
    subset=["customer_text"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=200000
)

matrix = vectorizer.fit_transform(
    conversations["customer_text"]
)

joblib.dump(vectorizer, VECTOR_PATH)
joblib.dump(matrix, MATRIX_PATH)

conversations.to_csv(
    DOCUMENTS_PATH,
    index=False
)

print()
print("Leakage-Free Retrieval Index")
print("============================")
print()
print(f"Golden examples excluded: {len(golden_texts)}")
print(f"Remaining documents: {len(conversations)}")
print(f"Features: {len(vectorizer.get_feature_names_out())}")
print(f"Matrix shape: {matrix.shape}")
print()
print(f"Saved vectorizer: {VECTOR_PATH}")
print(f"Saved matrix: {MATRIX_PATH}")
print(f"Saved documents: {DOCUMENTS_PATH}")