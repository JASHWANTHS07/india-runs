import pytest
from pathlib import Path
from src.loader import load_candidates, load_sample

DATA_DIR = Path(r"H:\india_runs\Data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge")
SAMPLE_PATH = DATA_DIR / "sample_candidates.json"
JSONL_PATH = DATA_DIR / "candidates.jsonl"

def test_load_sample_returns_list():
    candidates = load_sample(SAMPLE_PATH)
    assert isinstance(candidates, list)
    assert len(candidates) == 50

def test_load_sample_has_required_keys():
    candidates = load_sample(SAMPLE_PATH)
    c = candidates[0]
    for key in ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"]:
        assert key in c, f"Missing key: {key}"

def test_load_candidates_streaming():
    count = 0
    for c in load_candidates(JSONL_PATH):
        assert "candidate_id" in c
        count += 1
        if count >= 5:
            break
    assert count == 5
