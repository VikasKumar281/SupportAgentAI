"""
Two baselines, as required by the assignment ("results vs. at least two
baselines: a trivial one and a simple one"):

  1. TrivialBaseline    -> always predicts the majority class from training data.
  2. TfidfLogRegBaseline -> TF-IDF (1-2 grams) + multinomial Logistic Regression.
     This also doubles as our "simple model" fallback intent classifier when
     no LLM API key is configured (see classify_llm.py).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_SEED
from src.intents import silver_label


class TrivialBaseline:
    """Predicts the single most frequent label seen in training. No features used."""

    def fit(self, texts, labels):
        self.majority_ = pd.Series(labels).value_counts().idxmax()
        return self

    def predict(self, texts):
        return [self.majority_] * len(texts)

    def predict_proba_top(self, texts):
        return [1.0] * len(texts)  # "confidence" is meaningless here; kept for interface symmetry


class TfidfLogRegBaseline:
    """Simple, fast, fully offline intent classifier."""

    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0, random_state=RANDOM_SEED)),
        ])

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict(self, texts):
        return list(self.pipe.predict(texts))

    def predict_proba_top(self, texts):
        proba = self.pipe.predict_proba(texts)
        return list(np.max(proba, axis=1))

    def predict_with_confidence(self, texts):
        preds = self.predict(texts)
        confs = self.predict_proba_top(texts)
        return list(zip(preds, confs))

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        obj = cls()
        obj.pipe = joblib.load(path)
        return obj


def make_silver_training_labels(conversations_df: pd.DataFrame) -> pd.Series:
    """
    Weak-supervision labels for training (see src/intents.py docstring for why
    we don't train directly on the golden set). For our synthetic data we
    *could* cheat and use `true_intent`, but we deliberately use the same
    silver labeler a real deployment would have to rely on, so reported
    numbers reflect a realistic (imperfect) training signal, not an oracle.
    """
    return conversations_df["customer_text"].apply(silver_label)


def train_and_save(conversations_df: pd.DataFrame):
    labels = make_silver_training_labels(conversations_df)
    model = TfidfLogRegBaseline().fit(conversations_df["customer_text"].tolist(), labels.tolist())
    model.save()

    trivial = TrivialBaseline().fit(conversations_df["customer_text"].tolist(), labels.tolist())
    return model, trivial


if __name__ == "__main__":
    from src.config import CONVERSATIONS_CSV
    df = pd.read_csv(CONVERSATIONS_CSV)
    model, trivial = train_and_save(df)
    print(f"Trained on {len(df)} silver-labeled examples. Majority class: {trivial.majority_}")
    print(f"Saved TF-IDF+LogReg model to {MODEL_PATH}")
