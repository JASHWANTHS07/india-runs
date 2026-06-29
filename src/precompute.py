"""
GPU pre-computation: embed all candidates, compute BM25, cross-encoder re-rank top-N.

Outputs to artifacts/:
  - embeddings.npy       (N, 1024) float32 — bi-encoder embeddings
  - jd_embedding.npy     (1024,)   float32 — JD embedding
  - features.parquet     N rows    — structured features + bm25_score + cross_encoder_score

Usage:
    python src/precompute.py \
        --candidates candidates.jsonl \
        --artifacts ./artifacts
"""

import argparse
import dataclasses
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.loader import load_candidates
from src.features import extract_features
from src.bm25 import compute_bm25_scores

# Focused JD query for embedding — signal-rich, no boilerplate
JD_QUERY = (
    "Senior AI Engineer with 5-9 years production experience at product companies. "
    "Required: production embeddings-based retrieval using sentence-transformers, BGE, E5; "
    "vector databases Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch; "
    "strong Python; ranking evaluation frameworks NDCG, MRR, MAP; "
    "shipped search or recommendation or ranking systems to real users at scale. "
    "Nice to have: LLM fine-tuning LoRA QLoRA PEFT, learning-to-rank XGBoost neural. "
    "Disqualify: consulting-only career TCS Infosys Wipro Accenture, pure research academic, "
    "CV speech robotics without NLP information retrieval. "
    "Location: India Pune Noida Bangalore Hyderabad Mumbai Delhi preferred."
)

BI_ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32
CROSS_ENCODER_TOP_N = 1000  # Re-rank top-N by bi-encoder score


def main(candidates_path: str, artifacts_dir: str) -> None:
    t0 = time.time()
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load & extract features ----
    print(f"Loading candidates from {candidates_path}...")
    all_features = []
    all_texts = []
    for candidate in tqdm(load_candidates(candidates_path), desc="Extracting features"):
        feat = extract_features(candidate)
        all_features.append(feat)
        all_texts.append(feat.profile_text)
    N = len(all_features)
    print(f"  {N} candidates loaded, features extracted")

    # ---- BM25 sparse retrieval ----
    print("\nComputing BM25 scores...")
    t_bm25 = time.time()
    bm25_scores = compute_bm25_scores(all_texts, JD_QUERY)
    print(f"  BM25 done in {time.time() - t_bm25:.1f}s")

    # ---- Bi-encoder embeddings ----
    print(f"\nLoading bi-encoder: {BI_ENCODER_MODEL} (device: {device})")
    model = SentenceTransformer(BI_ENCODER_MODEL, device=device)

    print("Embedding JD query...")
    jd_emb = model.encode(JD_QUERY, normalize_embeddings=True)
    np.save(artifacts / "jd_embedding.npy", jd_emb.astype(np.float32))

    print(f"Embedding {N} candidates...")
    all_embeddings = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Bi-encoder batches"):
        batch = all_texts[i:i + BATCH_SIZE]
        embs = model.encode(batch, normalize_embeddings=True,
                            batch_size=BATCH_SIZE, show_progress_bar=False)
        all_embeddings.append(embs.astype(np.float32))
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    embeddings_array = np.vstack(all_embeddings)
    np.save(artifacts / "embeddings.npy", embeddings_array)
    print(f"  Embeddings saved: {embeddings_array.shape}")

    # Cosine similarities (embeddings are L2-normalized)
    semantic_scores = embeddings_array @ jd_emb.astype(np.float32)

    # ---- Cross-encoder re-ranking (top-N by bi-encoder) ----
    print(f"\nCross-encoder re-ranking top-{CROSS_ENCODER_TOP_N}...")
    t_ce = time.time()

    # Get top-N indices by bi-encoder score
    top_n_indices = np.argsort(-semantic_scores)[:CROSS_ENCODER_TOP_N]

    # Load cross-encoder
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

    # Build (query, candidate_text) pairs for top-N
    ce_pairs = [(JD_QUERY, all_texts[idx]) for idx in top_n_indices]

    # Score with cross-encoder
    ce_raw_scores = cross_encoder.predict(
        ce_pairs, batch_size=64, show_progress_bar=True
    )

    # Normalize cross-encoder scores to [0, 1]
    ce_min, ce_max = float(ce_raw_scores.min()), float(ce_raw_scores.max())
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

    cross_encoder_scores = np.zeros(N, dtype=np.float32)
    for i, idx in enumerate(top_n_indices):
        cross_encoder_scores[idx] = (float(ce_raw_scores[i]) - ce_min) / ce_range

    print(f"  Cross-encoder done in {time.time() - t_ce:.1f}s")

    # Free GPU memory
    del model, cross_encoder
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- Save features + new scores to parquet ----
    records = [dataclasses.asdict(f) for f in all_features]
    df = pd.DataFrame(records)
    df.drop(columns=["profile_text"], inplace=True)
    df["bm25_score"] = bm25_scores
    df["cross_encoder_score"] = cross_encoder_scores.tolist()
    df.to_parquet(artifacts / "features.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\nAll artifacts saved to {artifacts}/ ({elapsed:.1f}s total)")
    print(f"  embeddings.npy    : {embeddings_array.shape}")
    print(f"  jd_embedding.npy  : {jd_emb.shape}")
    print(f"  features.parquet  : {len(df)} rows, {len(df.columns)} columns")
    print(f"  (includes bm25_score + cross_encoder_score)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifacts", default="./artifacts")
    args = parser.parse_args()
    main(args.candidates, args.artifacts)
