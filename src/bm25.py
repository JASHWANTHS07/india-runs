"""
BM25 sparse retrieval scoring.

Computes BM25 similarity between each candidate's profile text and the JD query.
Runs on CPU, no external dependencies beyond rank_bm25.
"""

import re
import math
from typing import List


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


class BM25:
    """Okapi BM25 scorer (k1=1.5, b=0.75)."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / max(1, self.N)

        # Document frequencies
        self.df = {}
        for doc in corpus_tokens:
            seen = set(doc)
            for term in seen:
                self.df[term] = self.df.get(term, 0) + 1

        # Pre-compute IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

        # Store doc lengths and term frequencies
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.doc_tfs = []
        for doc in corpus_tokens:
            tf = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.doc_tfs.append(tf)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Score a single document against a query."""
        tf_dict = self.doc_tfs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_dict.get(term, 0)
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def score_all(self, query_tokens: List[str]) -> List[float]:
        """Score all documents against a query. Returns list of scores."""
        return [self.score(query_tokens, i) for i in range(self.N)]


def compute_bm25_scores(candidate_texts: List[str], jd_query: str) -> List[float]:
    """
    Compute BM25 scores for all candidates against the JD query.
    
    Returns normalized scores in [0, 1].
    """
    corpus_tokens = [_tokenize(text) for text in candidate_texts]
    query_tokens = _tokenize(jd_query)

    bm25 = BM25(corpus_tokens)
    raw_scores = bm25.score_all(query_tokens)

    # Normalize to [0, 1]
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score > 0:
        return [s / max_score for s in raw_scores]
    return raw_scores
