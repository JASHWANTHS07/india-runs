# India Runs — AI Recruiter Candidate Ranking System

## Project Overview

Redrob Hackathon submission by Jashwanth Sangu (solo). Rank 100,000 candidates against a fixed JD (Senior AI Engineer, Redrob AI) and produce a top-100 CSV with scores and reasoning.

**Deadline:** July 1, 2026 (tomorrow night). 3 submissions max — none used yet.

**Scoring formula:**
```
Composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**Hard constraints (ranking step only):**
| Constraint | Limit | Our target |
|---|---|---|
| Wall-clock | ≤ 5 min | < 90 sec |
| RAM | ≤ 16 GB | < 4 GB |
| GPU | None | None in `rank.py` |
| Network | Off | Zero API calls |
| Disk | ≤ 5 GB | < 1 GB |
| Honeypots in top 100 | > 10% = DQ | 0% |

## 5-Stage Elimination Pipeline

1. **Format validation** — auto-validator checks CSV shape/header/ranks/ids
2. **Scoring** — composite computed against hidden ground truth (no feedback)
3. **Code reproduction** — sandboxed Docker: 5min/16GB/CPU/no-network; honeypot rate > 10% = DQ
4. **Manual review** — 10 random reasoning rows checked for: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency
5. **Defend interview** — 30-min call with Redrob engineering

## Architecture

Two-phase split:

### Offline (precompute.py, unbounded time, GPU/network OK)
1. Load 100K candidates from JSONL
2. Extract 76 structured features per candidate → `features.parquet`
3. Compute BM25 scores against JD query
4. Embed all candidates with `BAAI/bge-large-en-v1.5` (1024-dim) → `embeddings.npy`
5. Cross-encoder re-rank top-1000 with `cross-encoder/ms-marco-MiniLM-L-6-v2`
6. Save artifacts to `artifacts/`

### Online (rank.py, scored, CPU-only)
1. Load precomputed artifacts
2. Compute semantic similarity (dot product on L2-normalized embeddings)
3. Compute heuristic scores via `scoring.py`
4. Train XGBoost LTR on pseudo-labels (rank:pairwise)
5. Score all candidates, apply honeypot zeroing
6. Sort, take top 100, generate reasoning
7. Write `submission.csv`

## File Map

```
india-runs/
├── CLAUDE.md                    # THIS FILE — project context
├── Readme.md                    # 700-line detailed README
├── run_pipeline.py              # Full pipeline orchestrator (precompute + rank)
├── requirements.txt             # Python dependencies
├── submission_metadata.yaml     # Hackathon portal metadata
├── Idea_Submission_Redrob.pdf   # Submitted slide deck
├── src/
│   ├── loader.py          (20L)  # JSONL/GZ streaming loader
│   ├── features.py       (673L)  # 76-field CandidateFeatures dataclass + extraction
│   ├── bm25.py            (85L)  # Custom Okapi BM25 scoring
│   ├── honeypot.py        (14L)  # 3-condition honeypot detection
│   ├── scoring.py        (132L)  # Multiplicative heuristic scoring engine
│   ├── reasoning.py      (305L)  # Rank-aware reasoning generation
│   ├── precompute.py     (161L)  # GPU precomputation (embeddings, BM25, cross-encoder)
│   └── rank.py           (681L)  # CPU ranking: XGBoost LTR + heuristic
├── tests/
│   ├── test_features.py  (105L)  # 10 tests — feature extraction
│   ├── test_honeypot.py   (32L)  # 4 tests — honeypot detection
│   ├── test_loader.py     (26L)  # 3 tests — data loading (BROKEN: Windows paths)
│   ├── test_reasoning.py  (66L)  # 6 tests — reasoning generation
│   ├── test_scoring.py   (110L)  # 11 tests — scoring (BROKEN: stale imports)
│   └── validate_submission.py    # Standalone CSV validator (not pytest)
├── artifacts/
│   ├── features.parquet          # Precomputed features for 100K candidates (gitignored)
│   ├── jd_embedding.npy          # JD embedding vector (gitignored)
│   └── tuned_params.json         # Optuna-tuned XGBoost hyperparams
├── dataset/India_runs_data_and_ai_challenge/
│   ├── candidates.jsonl          # 100K candidate profiles (~465MB)
│   ├── candidate_schema.json     # JSON Schema definition
│   ├── job_description.docx      # The fixed JD
│   ├── sample_candidates.json    # 50-candidate sample
│   └── sample_submission.csv     # Format reference (intentionally bad)
├── docs/
│   ├── submission_spec.docx      # Submission rules + scoring
│   └── redrob_signals_doc.docx   # 23 behavioral signals reference
├── submissions/                  # Generated CSVs (gitignored)
├── notebooks/
│   ├── redrob_pipeline.ipynb     # Colab notebook
│   └── redrob_kaggle.ipynb       # Kaggle notebook
└── data/                         # Empty placeholder
```

## JD Requirements — What the Hidden Ground Truth Likely Rewards

### Must-have (highest weight)
- Production embeddings-based retrieval (sentence-transformers, BGE, E5)
- Vector DB experience (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch)
- Strong Python
- Ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing)
- 5-9 years experience
- Product company background (not consulting-only)

### Nice-to-have
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank (XGBoost, neural)
- HR-tech/recruiting/marketplace exposure
- Distributed systems / large-scale inference
- Open-source contributions / GitHub activity

### Explicit disqualifiers (should rank near bottom)
1. Pure research only — no production deployment
2. AI experience = only recent (<12 months) LangChain+OpenAI calls
3. Senior engineers who haven't written production code in 18+ months
4. Title-chasers (Senior→Staff→Principal via 1.5yr hops)
5. "Framework enthusiasts" whose GitHub is LangChain tutorials
6. Entire career at consulting firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini)
7. Primary expertise in CV/speech/robotics without NLP/IR exposure
8. 5+ years entirely closed-source with zero external validation

### The JD's stated trap
> "The 'right answer' to this JD is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."
> "Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."

## Scoring Architecture (REVISED — multiplicative hierarchy)

### Design principle: Signal hierarchy
```
Tier 1 (MUST discriminate): retrieval_months, production_ml_depth, vector_db
Tier 2 (SHOULD discriminate): yoe_fit, skill_match, product_company
Tier 3 (MODIFIES only): behavioral, location, education
Tier 4 (FILTERS): honeypot, disqualifiers
```

### Score formula (multiplicative, not weighted average)
```python
score = relevance_gate × technical_core × fit_multiplier × behavioral × availability
```

Where:
```python
relevance = 0.35 * title + 0.45 * career_depth + 0.20 * coherence  # gate at 0.08

technical_core = (                    # PRIMARY DISCRIMINATOR
    0.35 * retrieval_depth +          # min(1.0, retrieval_months/60)
    0.25 * production_depth +         # min(1.0, ai_ml_months/72)
    0.15 * vector_db +                # 1.0 if has, else 0.0
    0.10 * shipped +                  # min(1.0, shipped_count/3)
    0.15 * semantic_sim               # embedding similarity
)

fit_multiplier = max(0.3, 0.50 * exp_fit + 0.50 * skills_fit)  # floor=0.3
behavioral = clamp(0.7, 1.2, behavioral_score(f))               # narrow range
availability = notice_factor * location_factor
```

### Hard penalties (softened YOE cliff)
| Condition | Multiplier |
|---|---|
| `technical_yoe < 5.0` | exp(-2 × (5-yoe)²), min 0.05 (gradual, not cliff) |
| `consulting_only + yoe > 3` | ×0.10 |
| `non_tech_title + AI skills + <12mo AI` | ×0.05 |
| `title_tier == 0` (non-tech) | ×0.20 |
| `cv_only_title + <6mo retrieval` | ×0.40 |
| Honeypot | return 0.0 |

### Title tiers
| Tier | Score | Examples |
|---|---|---|
| T4 (1.00) | Perfect | ML/AI/NLP/Search/Data Scientist Engineer |
| T3 (0.60) | Adjacent | SWE, Backend, Data, Platform Engineer |
| T2 (0.25) | Weak | Junior ML, DevOps, Frontend, QA, Mobile |
| T1 (0.08) | Manager | Catch-all for "engineer"/"developer" not in above |
| T0 (0.02) | Non-tech | HR, Marketing, Sales, Accountant, Content Writer |

### Experience fit curve
Gaussian centered at 7.0 years, sigma=1.5. Below 5: max(0.05, yoe/5 × 0.3). Above 9: linear decay.

## Next Steps (before submission)

1. **Re-run precompute** on Colab/Kaggle to regenerate `features.parquet` with 12 new columns
2. **Re-tune hyperparams**: `python src/rank.py --artifacts ./artifacts --out ./jashwanth_s.csv --method xgboost --tune --tune-trials 50`
3. **Validate**: `python tests/validate_submission.py jashwanth_s.csv`
4. **Inspect top-10** manually for retrieval depth ordering and reasoning quality
5. **Submit** (3 attempts available)

## Key Constants (in `features.py`)

- `REFERENCE_DATE`: datetime(2026, 6, 27) — anchor for all recency calculations
- `CONSULTING_FIRMS`: 16 Indian IT firms (TCS, Infosys, Wipro, etc.)
- `JD_SKILLS_TIER1`: 27 core skills (embeddings, vector DBs, retrieval, ranking)
- `JD_SKILLS_TIER2`: 26 nice-to-have skills (PyTorch, LLMs, MLOps, NLP)
- `INDIA_PREFERRED_CITIES`: Pune, Noida, Delhi, Bangalore, Hyderabad, Mumbai, Chennai
- `NOTABLE_COMPANIES`: Google, Microsoft, Amazon, Meta, Netflix, Uber, LinkedIn, Apple, Flipkart, Salesforce
- `PROFICIENCY_WEIGHTS`: expert=1.0, advanced=0.75, intermediate=0.5, beginner=0.25

## Honeypot Detection (REVISED — 8 conditions)

Any condition → honeypot (return 0.0 score):
1. `timeline_impossible` — stated YOE exceeds career history by > 5 years
2. `expert_zero_usage_count > 3` — 4+ skills "expert" with 0 months + 0 endorsements
3. `expert_zero_usage_count >= 2 AND skill_career_coherence < 0.15`
4. **NEW:** `salary_inverted` — salary min > max (26% of sample candidates!)
5. **NEW:** `assessment_proficiency_gap > 2.0 AND assessment_count >= 2` — claims expert, scores <30
6. **NEW:** `summary_is_template AND career_desc_title_mismatch >= 2`
7. **NEW:** `endorsements > 5×connections AND connections < 10`
8. **NEW:** Ghost profile: completeness≥80, inactive>180d, 0 applications, 0 views

## XGBoost LTR Pipeline (in `rank.py`)

1. Level 1 hard filter: `technical_yoe >= 5.0`
2. Pseudo-labels (0-4 graded):
   - Grade 4: T4 title + retrieval≥24mo + product AI + shipped + India
   - Grade 3: T3+ + AI≥24mo + product + retrieval≥12mo
   - Grade 2: T3+ + AI≥12mo + retrieval≥6mo (or T4 + coherence≥0.3)
   - Grade 1: T2+ + AI≥6mo
   - Grade 0: honeypot, consulting-only, non-tech+low-coherence
3. XGBoost rank:pairwise, 67 features, Optuna-tuned hyperparams
4. Score normalization: percentile-based on qualified pool
5. Post: honeypots zeroed, <5yr zeroed

### Tuned hyperparams (artifacts/tuned_params.json)
```
lr=0.077, max_depth=4, min_child_weight=6, subsample=0.719,
colsample_bytree=0.975, gamma=1.088, lambda=0.628, alpha=0.419
n_rounds=19, val_ndcg10=1.0
```

## Reasoning Generation (in `reasoning.py`)

Three-part structure: lead sentence + JD connection + supports/concerns.

Lead priority: notable company > tier-1 education > deep retrieval > multi-shipped > ML certs > deep AI > has retrieval > has shipped > fallback.

Rank-tier prefixes: 90+ = "Borderline fit:", 70+ = "Adequate but not strong:", else positive.

## Submission Output Format

```csv
candidate_id,rank,score,reasoning
CAND_XXXXXXX,1,0.95,"Feature-grounded reasoning string"
```

- Exactly 100 data rows, ranks 1-100
- Scores non-increasing with rank
- Ties broken by candidate_id ascending
- Validate with: `python tests/validate_submission.py submission.csv`

## Known Issues (as of 2026-06-30)

### Broken Tests — FIXED (2026-06-30)
All test imports fixed, conftest.py created with shared fixture, Windows paths replaced. 39/39 tests pass.

### Missing Components
- No `app.py` sandbox demo (README references one)
- No `.github/workflows/ci.yml`
- `embeddings.npy` not in artifacts/ (gitignored, must be on Colab/Kaggle)
- **MUST re-run precompute** on Colab/Kaggle to generate features.parquet with 12 new columns

### Code Quality — FIXED (2026-06-30)
- `NOTABLE_COMPANIES` deduped — reasoning.py now imports from features.py
- `val_ndcg10: 1.0` — pseudo-labels improved with cross-encoder signal to reduce circularity
- **Must re-tune hyperparams** after precompute re-run (`--tune --tune-trials 50`)

### Scoring Concerns (being fixed in v2 overhaul)
- Top-100 score range 1.0→0.85 — fix: sigmoid normalization instead of percentile
- `technical_yoe < 5` hard cliff (0.05×) — fix: gradual Gaussian decay
- Retrieval experience underweighted — fix: 35% of technical_core (was ~12% of final)
- Flat weighted average lets behavioral override technical — fix: multiplicative hierarchy

## Development Workflow

### Full pipeline (requires GPU)
```bash
python run_pipeline.py --candidates dataset/India_runs_data_and_ai_challenge/candidates.jsonl --artifacts ./artifacts --out ./jashwanth_s.csv --method xgboost
```

### Ranking only (CPU, uses existing artifacts)
```bash
python src/rank.py --artifacts ./artifacts --out ./jashwanth_s.csv --method xgboost
```

### With Optuna tuning
```bash
python src/rank.py --artifacts ./artifacts --out ./jashwanth_s.csv --method xgboost --tune --tune-trials 50
```

### Validate
```bash
python tests/validate_submission.py jashwanth_s.csv
```

### Run tests
```bash
cd /Users/Jashwanth.S/india-runs && python -m pytest tests/ -v
```

## Compute Environment

- **Precompute:** Google Colab T4 GPU / Kaggle 2× T4
- **Ranking:** CPU-only (any machine with Python 3.10+, 4GB RAM)
- **Dependencies:** sentence-transformers, numpy, pandas, pyarrow, xgboost, lightgbm, optuna, pytest

## Submission Metadata

- **Team:** "Jashwanth S" (solo)
- **Contact:** jashwanthsangu14@gmail.com
- **GitHub:** https://github.com/JASHWANTHS07/india-runs
- **AI tools declared:** Claude (Anthropic) — architecture design, code review, scoring iteration. No LLM at runtime.
