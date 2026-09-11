"""
Retrieval over the "how has this brand historically resolved similar issues"
knowledge base. We use TF-IDF cosine similarity rather than embeddings:
it's fully offline, deterministic, fast enough for a take-home, and
transparent (you can literally point at the matched words). The report
discusses this trade-off vs. embeddings in "what we chose not to build".
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalResolutionRetriever:
    def __init__(self, conversations_df: pd.DataFrame):
        self.df = conversations_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text"].tolist())

    def top_k(self, query_text: str, k: int = 3, intent_filter: str = None):
        """
        Returns list of dicts: {customer_text, agent_reply, similarity}
        sorted by similarity desc. If intent_filter given and the df has a
        'true_intent' or 'silver_intent' column, restrict the search to
        that intent first (falls back to unfiltered search if too few match).
        """
        candidate_idx = np.arange(len(self.df))
        if intent_filter is not None:
            for col in ("true_intent", "silver_intent"):
                if col in self.df.columns:
                    mask = self.df[col] == intent_filter
                    if mask.sum() >= k:
                        candidate_idx = np.where(mask.values)[0]
                    break

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix[candidate_idx]).flatten()
        order = np.argsort(-sims)[:k]

        results = []
        for i in order:
            row_idx = candidate_idx[i]
            row = self.df.iloc[row_idx]
            results.append({
                "customer_text": row["customer_text"],
                "agent_reply": row["agent_reply"],
                "similarity": float(sims[i]),
            })
        return results
