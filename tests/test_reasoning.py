import pytest
from src.features import CandidateFeatures
from src.reasoning import generate_reasoning

def test_reasoning_returns_nonempty_string(make_features):
    f = make_features()
    r = generate_reasoning(f, rank=1)
    assert isinstance(r, str) and len(r) > 20

def test_reasoning_references_actual_yoe(make_features):
    f = make_features(yoe=7.2, career_retrieval_months=0,
                       has_product_ai_career=False, shipped_count=0,
                       ai_ml_months=0, notable_company="",
                       best_education_tier=4, ml_cert_count=0)
    r = generate_reasoning(f, rank=1)
    assert "7.2" in r

def test_reasoning_references_actual_title(make_features):
    f = make_features(current_title="Senior ML Engineer")
    r = generate_reasoning(f, rank=1)
    assert "Senior ML Engineer" in r

def test_reasoning_mentions_concern_for_low_ranks(make_features):
    f = make_features(notice_period_days=120, recruiter_response_rate=0.1)
    r_low = generate_reasoning(f, rank=50)
    r_high = generate_reasoning(f, rank=5)
    # Concerns should appear at lower ranks
    assert "concern" in r_low.lower() or "notice" in r_low.lower()

def test_no_two_candidates_produce_identical_reasoning(make_features):
    f1 = make_features(candidate_id="CAND_0000001", yoe=7.0, top_matched_skill="FAISS")
    f2 = make_features(candidate_id="CAND_0000002", yoe=6.5, top_matched_skill="Python")
    r1 = generate_reasoning(f1, rank=1)
    r2 = generate_reasoning(f2, rank=2)
    assert r1 != r2

def test_reasoning_no_hallucination_no_made_up_company(make_features):
    f = make_features(current_company="Smallco")
    r = generate_reasoning(f, rank=1)
    assert "Google" not in r and "Amazon" not in r
