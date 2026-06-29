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
import json
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
    labels = np.ones(len(df), dtype=np.int32)  # Default: weak (grade 1)

    for i in range(len(df)):
        row = df.iloc[i]
        tier = int(row.get("title_relevance_tier", 0))
        retrieval_mo = int(row.get("career_retrieval_months", 0))
        ai_mo = int(row.get("ai_ml_months", 0))
        yoe = float(row.get("yoe", 0))
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

        # JD sweet spot: 5-9 years
        in_yoe_range = 5.0 <= yoe <= 9.0
        near_yoe_range = 4.0 <= yoe <= 11.0  # close enough

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
        # Too junior for Senior role — cap at grade 1
        if yoe < 3.0:
            labels[i] = 0
            continue

        # Grade 4: perfect fit — must be in YoE range
        if (tier == 4 and retrieval_mo >= 12 and has_product
                and "india" in country and shipped >= 1 and in_yoe_range):
            labels[i] = 4
            continue

        # Grade 3: strong fit — need at least near range
        if tier >= 3 and ai_mo >= 24 and has_product and near_yoe_range:
            labels[i] = 3
            if cross_enc >= 0.5 and in_yoe_range:
                labels[i] = 4  # Cross-encoder confirms + right YoE → perfect
            continue

        # Grade 2: moderate fit — YoE matters
        if tier >= 3 and ai_mo >= 12 and near_yoe_range:
            labels[i] = 2
            continue
        if tier == 4 and ai_mo < 12 and in_yoe_range:
            labels[i] = 2  # Right title + right YoE, light experience
            continue

        # Grade 1: weak — outside YoE range or weak title
        if tier >= 2 and ai_mo >= 6:
            # Penalize if far from YoE sweet spot
            if in_yoe_range:
                labels[i] = 2  # Boost: right YoE compensates for weaker title
            elif near_yoe_range:
                labels[i] = 1
            else:
                labels[i] = 1  # Too junior/senior stays weak
            continue

        # Everything else
        if tier <= 1:
            labels[i] = 0
        else:
            labels[i] = 1

    return labels


def tune_ltr_hyperparams(X: np.ndarray, y: np.ndarray, n_trials: int = 50,
                         n_folds: int = 5):
    """
    5-fold CV + Optuna Bayesian optimization for XGBoost ranking.

    Each trial:
      1. Suggests hyperparams
      2. For each fold: trains on 4 folds, evaluates NDCG@10 on held-out fold
         with early stopping to auto-find best n_rounds
      3. Returns mean NDCG@10 across folds

    Results are deterministic (seed=42) and saved to artifacts/tuned_params.json.
    """
    import xgboost as xgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create deterministic k-fold indices
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(y))
    fold_size = len(y) // n_folds
    folds = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < n_folds - 1 else len(y)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        folds.append((train_idx, val_idx))

    print(f"  {n_folds}-fold CV: ~{fold_size} samples per fold")

    # Pre-build DMatrices (saves time across trials)
    fold_data = []
    for train_idx, val_idx in folds:
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dtrain.set_group([len(train_idx)])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
        dval.set_group([len(val_idx)])
        fold_data.append((dtrain, dval))

    def objective(trial):
        params = {
            "objective": trial.suggest_categorical(
                "objective", ["rank:pairwise", "rank:ndcg"]
            ),
            "eval_metric": "ndcg@10",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "seed": 42,
            "verbosity": 0,
        }

        fold_scores = []
        fold_best_rounds = []

        for dtrain, dval in fold_data:
            # Train with early stopping — auto-finds best n_rounds
            model = xgb.train(
                params, dtrain,
                num_boost_round=500,  # max cap
                evals=[(dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            fold_best_rounds.append(model.best_iteration + 1)

            # Get best validation score
            val_result = model.eval_set(
                [(dval, "val")], iteration=model.best_iteration
            )
            ndcg = float(val_result.split(":")[1])
            fold_scores.append(ndcg)

            # Prune unpromising trials early (after fold 2)
            if len(fold_scores) >= 2:
                trial.report(np.mean(fold_scores), len(fold_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_ndcg = np.mean(fold_scores)
        # Store best n_rounds as user attribute for retrieval later
        trial.set_user_attr("best_n_rounds", int(np.median(fold_best_rounds)))
        trial.set_user_attr("fold_scores", [round(s, 5) for s in fold_scores])
        return mean_ndcg

    def progress_callback(study, trial):
        best_so_far = study.best_value
        status = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"{trial.value:.5f}"
        print(f"  Trial {trial.number + 1}/{n_trials}: {status}  "
              f"(best so far: {best_so_far:.5f})")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    best = study.best_params
    best_trial = study.best_trial
    n_rounds = best_trial.user_attrs["best_n_rounds"]
    fold_scores = best_trial.user_attrs["fold_scores"]

    print(f"\n  Best mean CV NDCG@10: {study.best_value:.5f}")
    print(f"  Per-fold scores: {fold_scores}")
    print(f"  Best n_rounds (median across folds): {n_rounds}")
    print(f"  Best params: {best}")
    print(f"  Trials completed: {len(study.trials)}, "
          f"pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    return best, n_rounds, study.best_value


def train_ltr_model(X: np.ndarray, y: np.ndarray, params: dict = None,
                    n_rounds: int = None):
    """Train XGBoost learning-to-rank model with pairwise objective."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] xgboost not installed, falling back to heuristic scoring")
        return None

    # All candidates belong to a single query group
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group([len(y)])
    print(f"  DMatrix: {X.shape[0]} samples, {X.shape[1]} features, group=[{len(y)}]")

    if params is None:
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@10",
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.1,
            "seed": 42,
            "verbosity": 0,
        }
    else:
        params = {**params, "eval_metric": "ndcg@10", "seed": 42, "verbosity": 0}

    if n_rounds is None:
        n_rounds = 100

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=25,
    )
    return model


def main(artifacts_dir: str, out_path: str, tune: bool = False,
         tune_trials: int = 50) -> None:
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

    # Hyperparameter tuning: tune → save, or load saved params
    best_params, best_rounds = None, None
    tuned_params_path = artifacts / "tuned_params.json"

    if tune:
        print(f"\nTuning XGBoost hyperparameters ({tune_trials} trials)...")
        best_params, best_rounds, best_ndcg = tune_ltr_hyperparams(
            X_np, y, n_trials=tune_trials
        )
        # Save to artifacts for future runs
        saved = {"params": best_params, "n_rounds": best_rounds,
                 "val_ndcg10": best_ndcg}
        with open(tuned_params_path, "w") as f:
            json.dump(saved, f, indent=2)
        print(f"  Saved tuned params to {tuned_params_path}")
    elif tuned_params_path.exists():
        with open(tuned_params_path) as f:
            saved = json.load(f)
        best_params = saved["params"]
        best_rounds = saved["n_rounds"]
        print(f"\nLoaded tuned params from {tuned_params_path} "
              f"(val NDCG@10: {saved.get('val_ndcg10', '?')})")

    # Train XGBoost LTR
    obj_name = best_params.get("objective", "rank:pairwise") if best_params else "rank:pairwise"
    print(f"Training XGBoost LTR ({obj_name})...")
    ltr_model = train_ltr_model(X_np, y, params=best_params, n_rounds=best_rounds)

    if ltr_model is not None:
        import xgboost as xgb
        dtest = xgb.DMatrix(X_np)
        ltr_scores = ltr_model.predict(dtest)

        # Normalize LTR scores to (0, 1) using rank-based percentile mapping
        # This avoids ties at 0.0/1.0 that min-max normalization creates
        order = ltr_scores.argsort()
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.float32)
        final_scores = ranks / (len(ranks) + 1)  # maps to (0, 1), no boundary ties

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
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    main(args.artifacts, args.out, tune=args.tune, tune_trials=args.tune_trials)
