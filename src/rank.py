"""
CPU ranking step — must complete in ≤5 min, no GPU, no network.

Usage:
    python src/rank.py \
        --artifacts ./artifacts \
        --out ./submission.csv

The --candidates argument is accepted for spec compatibility but not used at runtime
(all data is pre-computed in artifacts/).
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import fields

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import CandidateFeatures
from src.scoring import compute_score
from src.reasoning import generate_reasoning


_FEATURE_FIELDS = {f.name for f in fields(CandidateFeatures)}


def load_features(artifacts: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    embeddings = np.load(artifacts / "embeddings.npy")        # (N, D) float32
    jd_embedding = np.load(artifacts / "jd_embedding.npy")   # (D,) float32
    features_df = pd.read_parquet(artifacts / "features.parquet")
    return embeddings, jd_embedding, features_df


def df_row_to_features(row: pd.Series) -> CandidateFeatures:
    kwargs = {k: row[k] for k in _FEATURE_FIELDS if k in row.index}
    kwargs.setdefault("profile_text", "")
    return CandidateFeatures(**kwargs)


def main(artifacts_dir: str, out_path: str) -> None:
    t0 = time.time()
    artifacts = Path(artifacts_dir)
    out = Path(out_path)

    print("Loading pre-computed artifacts...")
    embeddings, jd_embedding, features_df = load_features(artifacts)
    print(f"  Loaded {len(features_df)} candidates, embeddings shape: {embeddings.shape}")

    # Cosine similarity — embeddings already L2-normalized, so dot product = cosine sim
    print("Computing semantic similarities...")
    semantic_scores = embeddings @ jd_embedding  # shape: (N,)

    # Score every candidate
    print("Computing final scores...")
    results = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        f = df_row_to_features(row)
        score = compute_score(f, float(semantic_scores[i]))
        results.append((score, f.candidate_id, f))

    # Sort descending by score, break ties by candidate_id ascending
    results.sort(key=lambda x: (-x[0], x[1]))

    # Guard: ensure we have at least 100 candidates
    if len(results) < 100:
        raise RuntimeError(f"Only {len(results)} candidates survived scoring; need at least 100.")

    # Take top 100
    top100 = results[:100]

    # Build submission rows
    output_rows = []
    for rank_idx, (score, cand_id, f) in enumerate(top100, start=1):
        reasoning = generate_reasoning(f, rank=rank_idx)
        output_rows.append({
            "candidate_id": cand_id,
            "rank": rank_idx,
            "score": round(float(score), 4),
            "reasoning": reasoning,
        })

    pd.DataFrame(output_rows).to_csv(out, index=False)
    elapsed = time.time() - t0
    print(f"Submission written to {out} ({elapsed:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None, help="Accepted for spec compat, not used")
    parser.add_argument("--artifacts", default="./artifacts")
    parser.add_argument("--out", default="./submission.csv")
    args = parser.parse_args()
    main(args.artifacts, args.out)
