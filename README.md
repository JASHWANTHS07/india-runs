# India Runs — AI Candidate Ranking System

**Redrob Hackathon** | Intelligent Candidate Discovery & Ranking Challenge

Rank 100,000 candidates against a Senior AI Engineer job description. Produces a top-100 CSV with scores and natural-language reasoning — no LLM at runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JASHWANTHS07/india-runs/blob/main/notebooks/redrob_pipeline.ipynb)

---

## Quick Start

### Option 1: Run on Google Colab (Recommended)

Click the badge above or open this link:

```
https://colab.research.google.com/github/JASHWANTHS07/india-runs/blob/main/notebooks/redrob_pipeline.ipynb
```

The notebook has two sections:

- **Section 1 — Sandbox Demo** (CPU, ~3 min): Runs the full pipeline on 50 sample candidates. No GPU needed. Demonstrates feature extraction, embedding, scoring, ranking, and reasoning generation.
- **Section 2 — Full Pipeline** (GPU, ~20 min): Runs on all 100K candidates with T4 GPU. Produces the actual submission CSV. Requires dataset extraction from `Data.rar`.

Just run Section 1 cells in order to see the pipeline work.

### Option 2: Run Locally

```bash
# Clone
git clone https://github.com/JASHWANTHS07/india-runs.git
cd india-runs

# Install dependencies
pip install -r requirements.txt

# Sample run (50 candidates, CPU, no GPU)
python run_pipeline.py \
    --run_sample dataset/India_runs_data_and_ai_challenge/sample_candidates.json \
    --artifacts ./artifacts_sample \
    --out ./sample_output.csv \
    --cpu \
    --method xgboost \
    --top-k 10
```

### Option 3: Ranking Only (with precomputed artifacts)

If you already have `artifacts/` from a prior precompute run:

```bash
python src/rank.py \
    --artifacts ./artifacts \
    --out ./jashwanth_s.csv \
    --method xgboost
```

Runs in <3 minutes on CPU. No GPU, no network.

---

## How It Works

### Two-Phase Architecture

```
candidates.jsonl (100K)
        │
        ▼
┌──────────────────────────────────────┐
│  GPU Phase (precompute.py, ~15 min)  │
│                                      │
│  Embedding: bge-large-en-v1.5        │
│  BM25: sparse keyword scoring        │
│  Cross-encoder: top-1000 re-rank     │
│  Feature extraction: 76 features     │
└──────────────┬───────────────────────┘
               │
        artifacts/
  (embeddings.npy, features.parquet)
               │
               ▼
┌──────────────────────────────────────┐
│  CPU Phase (rank.py, <3 min)         │
│                                      │
│  Cosine similarity (dot product)     │
│  XGBoost LTR (rank:pairwise)        │
│  Multiplicative scoring              │
│  Reasoning generation                │
└──────────────┬───────────────────────┘
               │
               ▼
        jashwanth_s.csv
   (top 100: id, rank, score, reasoning)
```

### Scoring Formula

Multiplicative hierarchy — signals don't average out, they gate each other:

```
score = relevance_gate x technical_core x fit_multiplier x behavioral x availability
```

| Stage | What it does | Weight breakdown |
|---|---|---|
| Relevance gate | Filters irrelevant titles/careers | 0.35 title + 0.45 career + 0.20 coherence |
| Technical core | Primary discriminator | 0.35 retrieval + 0.25 production + 0.15 vector_db + 0.10 shipped + 0.15 semantic |
| Fit multiplier | Experience + skill match | 0.50 exp_fit + 0.50 skills_fit (floor: 0.3) |
| Behavioral | Engagement signals | Clamped 0.7–1.2 |
| Availability | Location + notice period | notice_factor x location_factor |

### Honeypot Detection (8 Rules)

Catches ~19% of the 100K dataset as suspicious/synthetic profiles:

1. Impossible timelines (stated YOE exceeds career by 5+ years)
2. Expert-zero-usage (4+ skills "expert" with 0 months usage)
3. Expert-zero + low coherence
4. Inverted salary (min > max) — catches ~19% alone
5. Assessment-proficiency gap (claims expert, scores <30)
6. Template summary + career-title mismatch
7. Suspicious endorsement ratio (endorsements > 5x connections, connections < 10)
8. Ghost profile (high completeness, 180+ days inactive, 0 applications, 0 views)

---

## CLI Reference

### Full Pipeline

```bash
python run_pipeline.py \
    --candidates <path_to_candidates.jsonl> \
    --artifacts <artifacts_dir> \
    --out <output.csv> \
    --method xgboost \          # xgboost (recommended) or heuristic
    --top-k 100                 # number of candidates to output
```

### Sample/Demo Run

```bash
python run_pipeline.py \
    --run_sample <sample.json> \
    --artifacts <artifacts_dir> \
    --out <output.csv> \
    --cpu \
    --top-k 10
```

### Ranking Step Only

```bash
python src/rank.py \
    --artifacts <artifacts_dir> \
    --out <output.csv> \
    --method xgboost \
    --tune --tune-trials 50     # optional: Optuna hyperparameter tuning
```

### Validate Submission

```bash
python tests/validate_submission.py <submission.csv>
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
india-runs/
├── run_pipeline.py              # Full pipeline orchestrator
├── src/
│   ├── precompute.py            # GPU: embeddings + BM25 + cross-encoder + features
│   ├── features.py              # 76-field CandidateFeatures extraction
│   ├── scoring.py               # Multiplicative scoring engine (5 stages)
│   ├── reasoning.py             # NL reasoning with 5 lead variants
│   ├── honeypot.py              # 8-rule fraud detection
│   ├── rank.py                  # XGBoost LTR + heuristic blend → CSV output
│   ├── bm25.py                  # Custom Okapi BM25 scoring
│   └── loader.py                # JSONL/GZ streaming loader
├── tests/                       # 42 tests (pytest)
├── notebooks/
│   ├── redrob_pipeline.ipynb    # Colab notebook (sandbox + full pipeline)
│   └── redrob_kaggle.ipynb      # Kaggle notebook (dual T4 GPU)
├── artifacts/                   # Precomputed data (gitignored)
├── dataset/                     # 100K candidates + JD (gitignored)
├── docs/                        # Submission spec, signals doc
├── instructions.md              # Detailed planning document
├── submission_metadata.yaml     # Hackathon portal metadata
├── Idea_Submission_Redrob.pptx  # Submission slide deck
└── Idea_Submission_Redrob.pdf   # Submission slides (PDF)
```

---

## Compute Constraints

| Constraint | Limit | This System |
|---|---|---|
| Wall-clock (ranking) | 5 min | <3 min |
| RAM | 16 GB | <4 GB |
| GPU (ranking) | None | None |
| Network (ranking) | None | None |
| Disk | 5 GB | <1 GB |

---

## Tech Stack

- **Python 3.10+**
- **sentence-transformers** + BAAI/bge-large-en-v1.5 (1024-dim embeddings)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** (passage re-ranking)
- **XGBoost** (rank:pairwise LTR)
- **Optuna** (hyperparameter tuning)
- **NumPy / Pandas / PyArrow**
- **Google Colab / Kaggle** (GPU precompute)

---

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_XXXXXXX,1,0.95,"Feature-grounded reasoning string"
CAND_XXXXXXX,2,0.93,"Different reasoning string"
...
```

- 100 rows, ranks 1–100
- Scores non-increasing with rank
- Every reasoning string is unique and grounded in candidate data
- No LLM used — zero hallucination risk

---

## AI Tools Declaration

Claude (Anthropic) was used as an engineering co-pilot for architecture design, code review, and scoring formula iteration. No LLM is used at runtime — the pipeline is fully deterministic.

---

**Team:** Jashwanth S (Solo) | **Contact:** jashwanthsangu14@gmail.com | **GitHub:** [JASHWANTHS07/india-runs](https://github.com/JASHWANTHS07/india-runs)
