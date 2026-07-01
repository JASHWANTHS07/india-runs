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

from src.features import CandidateFeatures, _compute_title_tier, _is_cv_only_title
from src.scoring import compute_score
from src.reasoning import generate_reasoning
from src.honeypot import is_honeypot

_FEATURE_FIELDS = {f.name for f in fields(CandidateFeatures)}

# Columns used as XGBoost input features
LTR_FEATURE_COLS = [
    "yoe", "technical_yoe", "consulting_months", "product_months",
    "title_relevance_tier", "is_cv_only_title",
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
    # Education features
    "has_cs_degree", "highest_degree_level", "education_ai_relevance", "education_recency",
    # Certification features
    "cert_count", "ml_cert_count", "cert_recency",
    # Career trajectory features
    "ai_title_count", "title_progression", "avg_tenure_months", "num_roles",
    "max_company_size_ord", "current_company_size_ord",
    # Skill breadth features
    "total_skill_count", "avg_skill_proficiency", "endorsed_skill_ratio", "skill_keyword_density",
    # Work mode & platform features
    "work_mode_match", "search_appearance_30d", "salary_range_width", "platform_tenure_days",
    # New v2 features
    "headline_has_ai_keywords", "headline_has_generic_filler",
    "salary_inverted", "salary_fits_role",
    "assessment_count", "assessment_jd_count", "assessment_proficiency_gap",
    "market_demand_score",
    "summary_is_template", "summary_ai_keyword_count",
    "career_desc_title_mismatch_count", "career_production_keyword_density",
    # Reasoning-derived features (string fields excluded, bool proxy used)
    "has_notable_company",
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
    Domain-focused pseudo-relevance labels (0-4 graded relevance).

    Labels are computed for all candidates; the Level 1 hard filter
    (technical_yoe >= 5) selects which labels are actually used for
    XGBoost training. Labels focus on JD-specific domain fit:
    retrieval depth, AI career, title relevance, skill coherence,
    production experience.

    Grading:
      4 = Perfect domain fit (T4 title + deep retrieval + product AI + shipped)
      3 = Strong domain fit  (T4/T3 + significant AI + retrieval experience)
      2 = Moderate            (relevant title + some AI/retrieval)
      1 = Weak                (adjacent title, light domain signals)
      0 = Irrelevant          (non-tech, honeypot, no domain coherence)
    """
    labels = np.ones(len(df), dtype=np.int32)

    for i in range(len(df)):
        row = df.iloc[i]
        tier = int(row.get("title_relevance_tier", 0))
        retrieval_mo = int(row.get("career_retrieval_months", 0))
        ai_mo = int(row.get("ai_ml_months", 0))
        coherence = float(row.get("skill_career_coherence", 0))
        is_consult = bool(row.get("is_consulting_only", False))
        has_product = bool(row.get("has_product_ai_career", False))
        timeline_imp = bool(row.get("timeline_impossible", False))
        expert_zero = int(row.get("expert_zero_usage_count", 0))
        shipped = int(row.get("shipped_count", 0))
        cross_enc = float(row.get("cross_encoder_score", 0))
        country = str(row.get("country", "")).lower()
        sal_inv = bool(row.get("salary_inverted", False))
        is_template = bool(row.get("summary_is_template", False))
        desc_mismatch = int(row.get("career_desc_title_mismatch_count", 0))

        # Grade 0: clear irrelevance + new honeypot signals
        if timeline_imp or expert_zero > 3 or sal_inv:
            labels[i] = 0
            continue
        if is_template and desc_mismatch >= 2:
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

        # Grade 4: perfect domain fit
        if (tier >= 3 and retrieval_mo >= 12 and has_product
                and shipped >= 1 and "india" in country):
            labels[i] = 4
            continue

        # Grade 3: strong domain fit
        if tier >= 3 and ai_mo >= 12 and has_product and retrieval_mo >= 3:
            labels[i] = 3
            continue
        if tier >= 3 and retrieval_mo >= 12 and shipped >= 1:
            labels[i] = 3
            continue

        # Grade 2: moderate — relaxed to capture more signal
        if tier >= 3 and ai_mo >= 6:
            labels[i] = 2
            continue
        if tier >= 2 and ai_mo >= 12 and coherence >= 0.3:
            labels[i] = 2
            continue
        if tier == 4 and coherence >= 0.25:
            labels[i] = 2
            continue

        # Grade 1: weak
        if tier >= 2 and ai_mo >= 6:
            labels[i] = 1
            continue

        # Everything else
        if tier <= 1:
            labels[i] = 0
        else:
            labels[i] = 1

    return labels


def _build_folds(X, y, n_folds=5):
    """Create deterministic k-fold train/val index splits."""
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
    return folds, fold_size


def _make_groups(n, max_group_size=5000):
    """Split n samples into equal-ish groups under max_group_size."""
    n_groups = max(1, (n + max_group_size - 1) // max_group_size)
    base = n // n_groups
    remainder = n % n_groups
    # First 'remainder' groups get base+1, rest get base
    return [base + 1] * remainder + [base] * (n_groups - remainder)


def _train_lgbm_fold(params, X_tr, y_tr, X_val, y_val):
    """Train one LightGBM fold, return (best_iteration, val_ndcg@10)."""
    import lightgbm as lgb

    train_groups = _make_groups(len(y_tr))
    val_groups = _make_groups(len(y_val))
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=train_groups)
    dval = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=dtrain)

    callbacks = [
        lgb.early_stopping(20, verbose=False),
        lgb.log_evaluation(period=0),  # silent
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    best_iter = model.best_iteration
    best_score = model.best_score["val"]["ndcg@10"]
    return best_iter, best_score, model


def _train_xgb_fold(params, X_tr, y_tr, X_val, y_val):
    """Train one XGBoost fold, return (best_iteration, val_ndcg@10)."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group([len(y_tr)])
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group([len(y_val)])

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    best_iter = model.best_iteration + 1
    val_result = model.eval_set([(dval, "val")], iteration=model.best_iteration)
    ndcg = float(val_result.split(":")[1])
    return best_iter, ndcg, model


def tune_ltr_hyperparams(X: np.ndarray, y: np.ndarray, n_trials: int = 50,
                         n_folds: int = 5):
    """
    5-fold CV + Optuna Bayesian optimization across XGBoost AND LightGBM.

    Searches both frameworks in a single study — Optuna picks the winner.
    Early stopping auto-finds best n_rounds per fold. Pruning kills bad
    trials after 2 folds to save time.

    Results saved to artifacts/tuned_params.json.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Check which frameworks are available
    has_lgbm = False
    try:
        import lightgbm  # noqa: F401
        has_lgbm = True
    except ImportError:
        pass

    has_xgb = False
    try:
        import xgboost  # noqa: F401
        has_xgb = True
    except ImportError:
        pass

    if not has_xgb and not has_lgbm:
        print("  [WARN] Neither xgboost nor lightgbm installed, cannot tune")
        return None, None, 0.0

    # LightGBM's lambdarank requires sub-grouping for >10K samples, which
    # breaks global ranking (only optimizes within sub-groups). XGBoost
    # handles a single 100K group natively, so we only tune XGBoost.
    frameworks = ["xgboost"]
    print(f"  Frameworks: {frameworks}")

    folds, fold_size = _build_folds(X, y, n_folds)
    print(f"  {n_folds}-fold CV: ~{fold_size} samples per fold")

    def objective(trial):
        framework = trial.suggest_categorical("framework", frameworks)

        if framework == "lightgbm":
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": [10],
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
                "lambdarank_truncation_level": trial.suggest_int(
                    "lambdarank_truncation_level", 10, 50
                ),
                "seed": 42,
                "verbose": -1,
            }
            train_fold_fn = _train_lgbm_fold
        else:  # xgboost
            xgb_obj = trial.suggest_categorical(
                "xgb_objective", ["rank:pairwise", "rank:ndcg"]
            )
            params = {
                "objective": xgb_obj,
                "eval_metric": "ndcg@10",
                "eta": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "seed": 42,
                "verbosity": 0,
            }
            train_fold_fn = _train_xgb_fold

        fold_scores = []
        fold_best_rounds = []

        for train_idx, val_idx in folds:
            best_iter, ndcg, _ = train_fold_fn(
                params, X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            fold_best_rounds.append(best_iter)
            fold_scores.append(ndcg)

            # Prune unpromising trials after fold 2
            if len(fold_scores) >= 2:
                trial.report(np.mean(fold_scores), len(fold_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_ndcg = np.mean(fold_scores)
        trial.set_user_attr("best_n_rounds", int(np.median(fold_best_rounds)))
        trial.set_user_attr("fold_scores", [round(s, 5) for s in fold_scores])
        return mean_ndcg

    def progress_callback(study, trial):
        best_so_far = study.best_value
        if trial.state == optuna.trial.TrialState.PRUNED:
            status = "pruned"
        else:
            status = f"{trial.value:.5f}"
        fw = trial.params.get("framework", "?")
        print(f"  Trial {trial.number + 1}/{n_trials} [{fw}]: {status}  "
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
    print(f"  Winner: {best['framework']}")
    print(f"  Per-fold scores: {fold_scores}")
    print(f"  Best n_rounds (median across folds): {n_rounds}")
    print(f"  Best params: {best}")
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Trials: {completed} completed, {pruned} pruned")

    return best, n_rounds, study.best_value


def train_ltr_model(X: np.ndarray, y: np.ndarray, params: dict = None,
                    n_rounds: int = None, feature_names: list = None):
    """Train final LTR model (XGBoost or LightGBM) on all data."""
    framework = "xgboost"  # default
    if params and params.get("framework") == "lightgbm":
        framework = "lightgbm"

    if framework == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            print("  [WARN] lightgbm not installed, falling back to xgboost")
            framework = "xgboost"

    if framework == "lightgbm":
        import lightgbm as lgb
        # Build LightGBM params from tuned params
        lgb_params = {k: v for k, v in params.items()
                      if k not in ("framework", "xgb_objective")}
        if "lr" in lgb_params:
            lgb_params["learning_rate"] = lgb_params.pop("lr")
        lgb_params.setdefault("objective", "lambdarank")
        lgb_params.setdefault("metric", "ndcg")
        lgb_params.setdefault("eval_at", [10])
        lgb_params.setdefault("verbose", -1)
        lgb_params.setdefault("seed", 42)

        groups = _make_groups(len(y))
        dtrain = lgb.Dataset(X, label=y, group=groups,
                             feature_name=feature_names or "auto")
        if n_rounds is None:
            n_rounds = 100

        print(f"  LightGBM LambdaRank: {X.shape[0]} samples, "
              f"{X.shape[1]} features, {n_rounds} rounds")

        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[lgb.log_evaluation(25)],
        )
        return model

    # XGBoost path
    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] xgboost not installed, falling back to heuristic scoring")
        return None

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group([len(y)])
    print(f"  DMatrix: {X.shape[0]} samples, {X.shape[1]} features, group=[{len(y)}]")

    if params is None:
        xgb_params = {
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
        xgb_params = {k: v for k, v in params.items()
                      if k not in ("framework", "xgb_objective")}
        if "lr" in xgb_params:
            xgb_params["eta"] = xgb_params.pop("lr")
        if "xgb_objective" in params:
            xgb_params["objective"] = params["xgb_objective"]
        xgb_params.setdefault("objective", "rank:pairwise")
        xgb_params.update({"eval_metric": "ndcg@10", "seed": 42, "verbosity": 0})

    if n_rounds is None:
        n_rounds = 100

    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=25,
    )
    return model


def main(artifacts_dir: str, out_path: str, method: str = "heuristic",
         tune: bool = False, tune_trials: int = 50,
         reclassify_titles: bool = False) -> None:
    t0 = time.time()
    artifacts = Path(artifacts_dir)
    out = Path(out_path)

    print(f"Loading precomputed artifacts... (method={method})")
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

    # Derive boolean features from string fields
    if "notable_company" in features_df.columns:
        features_df["has_notable_company"] = (features_df["notable_company"].fillna("").str.len() > 0).astype(int)
    else:
        features_df["has_notable_company"] = 0

    # Reclassify title tiers using current code (skip re-precompute)
    if reclassify_titles:
        print("Reclassifying title tiers...")
        new_tiers = features_df["current_title"].apply(_compute_title_tier)
        new_cv = features_df["current_title"].apply(_is_cv_only_title)
        changed = (new_tiers != features_df["title_relevance_tier"]).sum()
        features_df["title_relevance_tier"] = new_tiers
        features_df["is_cv_only_title"] = new_cv
        print(f"  {changed} candidates reclassified")

    # Compute heuristic scores
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

    if method == "xgboost":
        # ---- Level 1: Hard filter on technical_yoe ----
        tech_yoe = features_df["technical_yoe"] if "technical_yoe" in features_df.columns else features_df["yoe"]
        qualified_mask = tech_yoe >= 5.0
        n_qualified = qualified_mask.sum()
        print(f"Level 1 filter: {n_qualified}/{N} candidates with technical_yoe >= 5.0")

        # ---- Level 2: XGBoost LTR on qualified pool ----
        print("Building LTR feature matrix...")
        available_cols = [c for c in LTR_FEATURE_COLS if c in features_df.columns]
        X_all = features_df[available_cols].copy()
        for col in X_all.columns:
            if X_all[col].dtype == bool:
                X_all[col] = X_all[col].astype(int)
        X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)

        # Train only on qualified candidates
        X_qualified = X_all[qualified_mask].values.astype(np.float32)
        X_all_np = X_all.values.astype(np.float32)

        print("Creating pseudo-relevance labels (qualified pool only)...")
        y_all = create_pseudo_labels(features_df)
        y_qualified = y_all[qualified_mask.values]
        label_dist = {g: int((y_qualified == g).sum()) for g in sorted(set(y_qualified))}
        print(f"  Label distribution (qualified): {label_dist}")

        # Hyperparameter tuning
        best_params, best_rounds = None, None
        tuned_params_path = artifacts / "tuned_params.json"

        if tune:
            print(f"\nTuning XGBoost hyperparameters ({tune_trials} trials, 5-fold CV)...")
            best_params, best_rounds, best_ndcg = tune_ltr_hyperparams(
                X_qualified, y_qualified, n_trials=tune_trials
            )
            saved = {"params": best_params, "n_rounds": best_rounds,
                     "val_ndcg10": best_ndcg}
            with open(tuned_params_path, "w") as f:
                json.dump(saved, f, indent=2)
            print(f"  Saved tuned params to {tuned_params_path}")
        elif tuned_params_path.exists():
            with open(tuned_params_path) as f:
                saved = json.load(f)
            if saved.get("params"):
                best_params = saved["params"]
                best_rounds = saved["n_rounds"]
                print(f"\nLoaded tuned params (val NDCG@10: {saved.get('val_ndcg10', '?')})")

        print("Training XGBoost LTR (qualified pool)...")
        ltr_model = train_ltr_model(X_qualified, y_qualified, params=best_params,
                                    n_rounds=best_rounds, feature_names=available_cols)

        if ltr_model is not None:
            import xgboost as xgb
            # Predict on ALL candidates (qualified get real scores, unqualified get zeroed)
            dtest = xgb.DMatrix(X_all_np)
            ltr_scores = ltr_model.predict(dtest)

            # Min-max normalization on raw XGBoost scores (top-focused)
            qualified_indices = np.where(qualified_mask.values)[0]
            final_scores = np.zeros(N, dtype=np.float32)
            q_scores = ltr_scores[qualified_indices]
            q_min, q_max = q_scores.min(), q_scores.max()
            if q_max > q_min:
                q_final = (q_scores - q_min) / (q_max - q_min)
            else:
                q_final = np.full_like(q_scores, 0.5)
            final_scores[qualified_indices] = q_final

            # Zero out honeypots within qualified pool
            for i in qualified_indices:
                if is_honeypot(feature_objects[i]):
                    final_scores[i] = 0.0

            print(f"  LTR scores (qualified): min={q_final.min():.4f}, max={q_final.max():.4f}")

            # Feature importance
            importance = ltr_model.get_score(importance_type="gain")
            top_feats = sorted(importance.items(), key=lambda x: -x[1])[:10]
            print(f"\n  Top-10 LTR features by gain:")
            for fname, gain in top_feats:
                col_idx = int(fname.replace("f", ""))
                col_name = available_cols[col_idx] if col_idx < len(available_cols) else fname
                print(f"    {col_name}: {gain:.1f}")
        else:
            print("  [WARN] XGBoost failed, falling back to heuristic scoring")
            final_scores = np.array(heuristic_scores, dtype=np.float32)

    else:
        # ---- Original heuristic scoring path ----
        print("Using heuristic scoring (original method)...")
        final_scores = np.array(heuristic_scores, dtype=np.float32)

    # Hard gate: below JD minimum technical experience → zero
    for i, f in enumerate(feature_objects):
        if f.technical_yoe < 5.0:
            final_scores[i] = 0.0

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None, help="Accepted for spec compat, not used")
    parser.add_argument("--artifacts", default="./artifacts")
    parser.add_argument("--out", default="./jashwanth_s.csv")
    parser.add_argument("--method", default="heuristic",
                        choices=["heuristic", "xgboost"],
                        help="Ranking method: 'heuristic' (original) or 'xgboost' (LTR)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning (xgboost method only)")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--reclassify-titles", action="store_true",
                        help="Recompute title tiers at rank time (skip re-precompute)")
    args = parser.parse_args()
    main(args.artifacts, args.out, method=args.method,
         tune=args.tune, tune_trials=args.tune_trials,
         reclassify_titles=args.reclassify_titles)
