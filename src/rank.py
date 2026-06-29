"""
CPU ranking step — XGBoost learning-to-rank over precomputed features.

Must complete in <=5 min, no GPU, no network.

Architecture:
  1. Load precomputed artifacts (embeddings, features, BM25, cross-encoder scores)
  2. Compute semantic similarity (dot product)
  3. Build feature matrix from all signals
  4. Create pseudo-relevance labels from strong composite signals
  5. Train XGBoost LTR (rank:pairwise) on pseudo-labels
  6. Predict final scores, rank top 100, generate reasoning

Usage:
    python src/rank.py --artifacts ./artifacts --out ./jashwanth_s.csv
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
from src.honeypot import is_honeypot

_FEATURE_FIELDS = {f.name for f in fields(CandidateFeatures)}

# Columns used as XGBoost input features
LTR_FEATURE_COLS = [
    "yoe", "title_relevance_tier", "is_cv_only_title",
    "has_product_company", "is_consulting_only",
    "ai_ml_months", "shipped_count", "vector_search_experience",
    "career_retrieval_months", "career_ai_depth_ratio", "has_product_ai_career",
    "skills_match_score", "jd_skill_count", "jd_tier1_skill_count",
    "skill_career_coherence", "non_tech_title_with_ai_skills",
    "jd_skill_assessment_avg", "best_education_tier",
    "in_preferred_india_city", "willing_to_relocate", "open_to_work",
    "days_since_active", "recruiter_response_rate", "github_activity_score",
    "notice_period_days", "profile_completeness",
    "applications_30d", "interview_completion_rate",
    "saved_by_recruiters", "profile_views_30d", "avg_response_time_hours",
    "verified_count", "endorsements_total", "connection_count",
    "offer_acceptance_rate", "timeline_impossible", "expert_zero_usage_count",
    # Retrieval scores (added during precompute)
    "semantic_sim", "bm25_score", "cross_encoder_score",
    # Derived from heuristic scoring
    "heuristic_score",
]


def load_artifacts(artifacts: Path):
    embeddings = np.load(artifacts / "embeddings.npy")
    jd_embedding = np.load(artifacts / "jd_embedding.npy")
    features_df = pd.read_parquet(artifacts / "features.parquet")
    return embeddings, jd_embedding, features_df


def df_row_to_features(row: pd.Series) -> CandidateFeatures:
    kwargs = {k: row[k] for k in _FEATURE_FIELDS if k in row.index}
    kwargs.setdefault("profile_text", "")
    return CandidateFeatures(**kwargs)


def create_pseudo_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create pseudo-relevance labels (0-4 graded relevance) from strong signals.
    
    This is a pseudo-relevance feedback approach: we use high-confidence signals
    to create training labels, then let XGBoost learn feature interactions we miss
    with hand-tuned weights.
    
    Grading:
      4 = Perfect fit (T4 title + retrieval career + product company + India)
      3 = Strong fit  (T4/T3 title + significant AI career + product)
      2 = Moderate    (relevant title + some AI experience)
      1 = Weak        (adjacent title or weak signals)
      0 = Irrelevant  (non-tech, honeypot, stuffer, consulting-only)
    """
    labels = np.ones(len(df), dtype=np.float32)  # Default: weak

    for i in range(len(df)):
        row = df.iloc[i]
        tier = int(row.get("title_relevance_tier", 0))
        retrieval_mo = int(row.get("career_retrieval_months", 0))
        ai_mo = int(row.get("ai_ml_months", 0))
        coherence = float(row.get("skill_career_coherence", 0))
        is_consult = bool(row.get("is_consulting_only", False))
        has_product = bool(row.get("has_product_ai_career", False))
        in_india_city = bool(row.get("in_preferred_india_city", False))
        timeline_imp = bool(row.get("timeline_impossible", False))
        expert_zero = int(row.get("expert_zero_usage_count", 0))
        shipped = int(row.get("shipped_count", 0))
        semantic = float(row.get("semantic_sim", 0))
        cross_enc = float(row.get("cross_encoder_score", 0))
        country = str(row.get("country", "")).lower()

        # Grade 0: clear irrelevance
        if timeline_imp or expert_zero > 3:
            labels[i] = 0
            continue
        if is_consult and tier < 4:
            labels[i] = 0
            continue
        if tier == 0 and coherence < 0.2:
            labels[i] = 0
            continue
        if coherence < 0.10 and ai_mo < 6:
            labels[i] = 0
            continue

        # Grade 4: perfect fit
        if (tier == 4 and retrieval_mo >= 12 and has_product
                and "india" in country and shipped >= 1):
            labels[i] = 4
            continue

        # Grade 3: strong fit
        if tier >= 3 and ai_mo >= 24 and has_product:
            labels[i] = 3
            if cross_enc >= 0.5:
                labels[i] = 3.5  # Cross-encoder confirms strong match
            continue

        # Grade 2: moderate fit
        if tier >= 3 and ai_mo >= 12:
            labels[i] = 2
            continue
        if tier == 4 and ai_mo < 12:
            labels[i] = 2  # Right title, light experience
            continue

        # Grade 1: weak
        if tier >= 2 and ai_mo >= 6:
            labels[i] = 1
            continue

        # Everything else stays at default 1 or drops
        if tier <= 1:
            labels[i] = 0.5
        else:
            labels[i] = 1

    return labels


def train_ltr_model(X: np.ndarray, y: np.ndarray):
    """Train XGBoost learning-to-rank model with pairwise objective."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] xgboost not installed, falling back to heuristic scoring")
        return None

    # Use rank:pairwise for learning-to-rank
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train")],
        verbose_eval=50,
    )
    return model


def main(artifacts_dir: str, out_path: str) -> None:
    t0 = time.time()
    artifacts = Path(artifacts_dir)
    out = Path(out_path)

    print("Loading precomputed artifacts...")
    embeddings, jd_embedding, features_df = load_artifacts(artifacts)
    N = len(features_df)
    print(f"  {N} candidates, embeddings: {embeddings.shape}")

    # Semantic similarity (dot product on L2-normalized vectors)
    print("Computing semantic similarities...")
    semantic_scores = embeddings @ jd_embedding
    features_df["semantic_sim"] = semantic_scores

    # Ensure BM25 and cross-encoder columns exist (backward compat)
    if "bm25_score" not in features_df.columns:
        print("  [WARN] bm25_score not in artifacts, using 0")
        features_df["bm25_score"] = 0.0
    if "cross_encoder_score" not in features_df.columns:
        print("  [WARN] cross_encoder_score not in artifacts, using 0")
        features_df["cross_encoder_score"] = 0.0

    # Compute heuristic scores (used as a feature for LTR)
    print("Computing heuristic scores...")
    heuristic_scores = []
    feature_objects = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        f = df_row_to_features(row)
        sim = float(semantic_scores[i])
        h_score = compute_score(f, sim)
        heuristic_scores.append(h_score)
        feature_objects.append(f)
    features_df["heuristic_score"] = heuristic_scores

    # Build LTR feature matrix
    print("Building LTR feature matrix...")
    available_cols = [c for c in LTR_FEATURE_COLS if c in features_df.columns]
    X = features_df[available_cols].copy()

    # Convert booleans to int for XGBoost
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Fill NaN/inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    X_np = X.values.astype(np.float32)

    # Create pseudo-relevance labels
    print("Creating pseudo-relevance labels...")
    y = create_pseudo_labels(features_df)
    label_dist = {g: int((y == g).sum()) for g in sorted(set(y))}
    print(f"  Label distribution: {label_dist}")

    # Train XGBoost LTR
    print("Training XGBoost LTR (rank:pairwise)...")
    ltr_model = train_ltr_model(X_np, y)

    if ltr_model is not None:
        import xgboost as xgb
        dtest = xgb.DMatrix(X_np)
        ltr_scores = ltr_model.predict(dtest)

        # Normalize LTR scores to [0, 1] with good spread
        ltr_min, ltr_max = ltr_scores.min(), ltr_scores.max()
        ltr_range = ltr_max - ltr_min if ltr_max > ltr_min else 1.0
        final_scores = (ltr_scores - ltr_min) / ltr_range

        # Apply honeypot zero-out AFTER LTR (hard override)
        for i, f in enumerate(feature_objects):
            if is_honeypot(f):
                final_scores[i] = 0.0

        print(f"  LTR scores: min={final_scores.min():.4f}, max={final_scores.max():.4f}")
    else:
        # Fallback: use heuristic scores
        final_scores = np.array(heuristic_scores, dtype=np.float32)

    # Round to 4 decimals BEFORE sorting (so tie-break matches CSV output)
    final_scores_rounded = np.round(final_scores, 4)

    # Build results and sort
    results = []
    for i in range(N):
        results.append((
            final_scores_rounded[i],
            features_df.iloc[i]["candidate_id"],
            feature_objects[i],
            float(semantic_scores[i]),
        ))

    # Sort descending by score, then ascending by candidate_id for tie-break
    results.sort(key=lambda x: (-x[0], x[1]))

    if len(results) < 100:
        raise RuntimeError(f"Only {len(results)} candidates; need >= 100")

    # Take top 100
    top100 = results[:100]

    # Build submission
    output_rows = []
    for rank_idx, (score, cand_id, f, sim) in enumerate(top100, start=1):
        reasoning = generate_reasoning(f, rank=rank_idx, semantic_sim=sim)
        output_rows.append({
            "candidate_id": cand_id,
            "rank": rank_idx,
            "score": float(score),
            "reasoning": reasoning,
        })

    pd.DataFrame(output_rows).to_csv(out, index=False)
    elapsed = time.time() - t0
    print(f"\nSubmission written to {out} ({elapsed:.1f}s)")
    print(f"  Score range: {output_rows[0]['score']:.4f} → {output_rows[-1]['score']:.4f}")

    # Feature importance (if LTR was trained)
    if ltr_model is not None:
        importance = ltr_model.get_score(importance_type="gain")
        top_feats = sorted(importance.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 LTR features by gain:")
        for fname, gain in top_feats:
            col_idx = int(fname.replace("f", ""))
            col_name = available_cols[col_idx] if col_idx < len(available_cols) else fname
            print(f"    {col_name}: {gain:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None, help="Accepted for spec compat, not used")
    parser.add_argument("--artifacts", default="./artifacts")
    parser.add_argument("--out", default="./jashwanth_s.csv")
    args = parser.parse_args()
    main(args.artifacts, args.out)
