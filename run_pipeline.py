"""
run_pipeline.py — Full pipeline orchestrator for Redrob hackathon submission.

Steps:
  1. Install dependencies (optional, pass --install)
  2. Pre-compute  : GPU embed + BM25 + cross-encoder + features  (GPU required)
  3. Rank         : XGBoost LTR → top-100 jashwanth_s.csv        (CPU only, <5 min)

Usage on Google Colab (GPU runtime):
    !python run_pipeline.py \
        --candidates "dataset/India_runs_data_and_ai_challenge/candidates.jsonl" \
        --artifacts  "./artifacts" \
        --out        "./jashwanth_s.csv" \
        --install

Sample/sandbox mode (small candidate file, full pipeline):
    python run_pipeline.py \
        --run_sample "./sample_candidates.json" \
        --artifacts  "./artifacts_sample" \
        --out        "./sample_output.csv"

Skip precompute (if artifacts already exist):
    python run_pipeline.py \
        --candidates "candidates.jsonl" \
        --artifacts  "./artifacts" \
        --out        "./jashwanth_s.csv" \
        --skip-precompute
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {step} failed (exit {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] {step} — {elapsed:.1f}s")


def convert_sample_to_jsonl(sample_path: str, output_path: str) -> str:
    """
    Convert a sample JSON or JSONL file to a standardized JSONL file.
    Accepts:
      - .json  (single object or array of objects)
      - .jsonl (one JSON object per line)
    Returns path to the JSONL file.
    """
    sample = Path(sample_path)
    out = Path(output_path)

    if sample.suffix == ".jsonl":
        # Already JSONL — just verify it's valid and copy
        candidates = []
        with open(sample, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif sample.suffix == ".json":
        with open(sample, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            # Single candidate or wrapper object
            if "candidates" in data:
                candidates = data["candidates"]
            else:
                candidates = [data]
        else:
            raise ValueError(f"Unexpected JSON structure in {sample_path}")
    else:
        raise ValueError(f"Unsupported file format: {sample.suffix} (use .json or .jsonl)")

    # Write as JSONL
    with open(out, "w") as f:
        for cand in candidates:
            f.write(json.dumps(cand) + "\n")

    print(f"  Sample: {len(candidates)} candidates from {sample_path}")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob hackathon pipeline: precompute (GPU) → rank (CPU) → submission CSV"
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl or candidates.jsonl.gz (full 100K dataset)",
    )
    parser.add_argument(
        "--run_sample",
        default=None,
        help="Path to a small sample file (.json or .jsonl) for sandbox/demo mode. "
             "Runs the full pipeline on this sample instead of the full dataset.",
    )
    parser.add_argument(
        "--artifacts",
        default="./artifacts",
        help="Directory for pre-computed artifacts (default: ./artifacts)",
    )
    parser.add_argument(
        "--out",
        default="./jashwanth_s.csv",
        help="Output submission CSV path (default: ./jashwanth_s.csv)",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip pre-computation if artifacts already exist",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (no GPU). Useful for sandbox/demo.",
    )
    parser.add_argument(
        "--method",
        default="heuristic",
        choices=["heuristic", "xgboost"],
        help="Ranking method: 'heuristic' (original) or 'xgboost' (LTR)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run pip install of dependencies before starting",
    )
    args = parser.parse_args()

    python = sys.executable
    artifacts = Path(args.artifacts)

    # Force CPU if requested
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] Forced CPU-only mode (CUDA_VISIBLE_DEVICES='')")

    # Determine candidate source
    if args.run_sample:
        # Sample mode — convert to JSONL and use that
        sample_jsonl = str(artifacts / "_sample.jsonl")
        artifacts.mkdir(parents=True, exist_ok=True)
        candidates_path = convert_sample_to_jsonl(args.run_sample, sample_jsonl)
        skip_precompute = False
    elif args.candidates:
        candidates_path = args.candidates
        skip_precompute = args.skip_precompute
    else:
        parser.error("Either --candidates or --run_sample is required.")

    if args.install:
        run(
            [python, "-m", "pip", "install", "-r", "requirements.txt"],
            "Installing dependencies",
        )

    if skip_precompute:
        required = [
            artifacts / "embeddings.npy",
            artifacts / "jd_embedding.npy",
            artifacts / "features.parquet",
        ]
        missing = [str(f) for f in required if not f.exists()]
        if missing:
            print(f"[ERROR] --skip-precompute set but artifacts missing:\n  " + "\n  ".join(missing))
            sys.exit(1)
        print(f"[OK] Skipping pre-computation — artifacts exist at {artifacts}")
    else:
        run(
            [
                python, "src/precompute.py",
                "--candidates", candidates_path,
                "--artifacts", str(artifacts),
            ],
            "Step 1/2: Pre-computation (GPU — embeds + BM25 + cross-encoder + features)",
        )

    rank_cmd = [
        python, "src/rank.py",
        "--artifacts", str(artifacts),
        "--out", args.out,
        "--method", args.method,
    ]
    run(rank_cmd, f"Step 2/2: Ranking (CPU — {args.method} → top 100)")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Submission: {Path(args.out).resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
