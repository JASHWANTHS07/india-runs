"""
run_pipeline.py — Full pipeline orchestrator for Redrob hackathon submission.

Steps:
  1. Install dependencies (optional, pass --install)
  2. Pre-compute  : GPU embed 100K candidates + extract features  (GPU required)
  3. Rank         : CPU scoring → top-100 submission.csv          (CPU only, <5 min)

Usage on Google Colab (GPU runtime):
    # Upload this project folder and candidates.jsonl to Colab, then:
    !python run_pipeline.py \\
        --candidates "/content/candidates.jsonl" \\
        --artifacts  "/content/artifacts" \\
        --out        "/content/submission.csv" \\
        --install

Usage locally (after precompute already done):
    python run_pipeline.py \\
        --candidates "H:/india_runs/Data/.../candidates.jsonl" \\
        --artifacts  "./artifacts" \\
        --out        "./submission.csv" \\
        --skip-precompute
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob hackathon pipeline: precompute (GPU) → rank (CPU) → submission.csv"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--artifacts",
        default="./artifacts",
        help="Directory for pre-computed artifacts (default: ./artifacts)",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output submission CSV path (default: ./submission.csv)",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip pre-computation if artifacts already exist",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run pip install -r requirements.txt before starting",
    )
    args = parser.parse_args()

    python = sys.executable
    artifacts = Path(args.artifacts)

    if args.install:
        run([python, "-m", "pip", "install", "-r", "requirements.txt"], "Installing dependencies")

    if args.skip_precompute:
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
                "--candidates", args.candidates,
                "--artifacts", args.artifacts,
            ],
            "Step 1/2: Pre-computation (GPU — embeds 100K candidates)",
        )

    run(
        [
            python, "src/rank.py",
            "--artifacts", args.artifacts,
            "--out", args.out,
        ],
        "Step 2/2: Ranking (CPU — scores candidates, outputs top 100)",
    )

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Submission: {Path(args.out).resolve()}")
    print(f"{'='*60}\n")
    print("Next: upload submission.csv to the Redrob portal.")


if __name__ == "__main__":
    main()
