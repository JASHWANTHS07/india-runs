"""
GPU pre-computation: embed all 100K candidate profiles + extract structured features.
Run once before rank.py. Requires CUDA GPU.

Usage:
    python src/precompute.py \
        --candidates "H:/india_runs/Data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
        --artifacts ./artifacts
"""

import argparse
import dataclasses
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path when running as `python src/precompute.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce CUDA memory fragmentation (helps on small-VRAM GPUs like RTX 3050 4GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.loader import load_candidates
from src.features import extract_features

# Focused JD query — captures the signal-rich requirements without noise from the full JD text
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

BATCH_SIZE = 32   # RTX 3050 4GB VRAM: safe for bge-large with up to 512-token sequences
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def main(candidates_path: str, artifacts_dir: str) -> None:
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME} (device: {device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Embed JD query
    print("Embedding JD query...")
    jd_emb = model.encode(JD_QUERY, normalize_embeddings=True)
    np.save(artifacts / "jd_embedding.npy", jd_emb.astype(np.float32))
    print(f"JD embedding saved: shape {jd_emb.shape}")

    # Stream candidates, extract features, batch-embed
    all_features = []
    all_texts = []
    all_embeddings = []

    print(f"Processing candidates from {candidates_path}...")
    for candidate in tqdm(load_candidates(candidates_path)):
        feat = extract_features(candidate)
        all_features.append(feat)
        all_texts.append(feat.profile_text)

        if len(all_texts) == BATCH_SIZE:
            embs = model.encode(all_texts, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False)
            all_embeddings.append(embs.astype(np.float32))
            all_texts = []
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Flush remaining
    if all_texts:
        embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs.astype(np.float32))

    # Save embeddings
    embeddings_array = np.vstack(all_embeddings)
    np.save(artifacts / "embeddings.npy", embeddings_array)
    print(f"Embeddings saved: shape {embeddings_array.shape}")

    # Save features as parquet (drop profile_text — already embedded)
    records = [dataclasses.asdict(f) for f in all_features]
    df = pd.DataFrame(records)
    df.drop(columns=["profile_text"], inplace=True)
    df.to_parquet(artifacts / "features.parquet", index=False)
    print(f"Features saved: {len(df)} candidates → {artifacts / 'features.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--artifacts", default="./artifacts", help="Output directory for pre-computed files")
    args = parser.parse_args()
    main(args.candidates, args.artifacts)
