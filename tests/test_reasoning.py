import pytest
from src.features import CandidateFeatures
from src.reasoning import generate_reasoning

def _make_features(**overrides) -> CandidateFeatures:
    defaults = dict(
        candidate_id="CAND_0000001", yoe=7.0, current_title="ML Engineer",
        current_company="Smallco", has_product_company=True, is_consulting_only=False,
        ai_ml_months=36, shipped_count=2, vector_search_experience=True,
        top_matched_skill="FAISS", skills_match_score=0.6, best_education_tier=2,
        country="india", city="pune, india", in_preferred_india_city=True,
        willing_to_relocate=True, open_to_work=True, last_active_date="2026-06-20",
        days_since_active=7, recruiter_response_rate=0.8, github_activity_score=70.0,
        notice_period_days=30, expected_salary_min=25.0, expected_salary_max=40.0,
        profile_completeness=85.0, applications_30d=3, interview_completion_rate=0.9,
        timeline_impossible=False, expert_zero_usage_count=0, profile_text="...",
    )
    defaults.update(overrides)
    # Filter out any keys not in CandidateFeatures (e.g., rank_)
    valid_keys = {
        'candidate_id', 'yoe', 'current_title', 'current_company', 'has_product_company',
        'is_consulting_only', 'ai_ml_months', 'shipped_count', 'vector_search_experience',
        'top_matched_skill', 'skills_match_score', 'best_education_tier', 'country', 'city',
        'in_preferred_india_city', 'willing_to_relocate', 'open_to_work', 'last_active_date',
        'days_since_active', 'recruiter_response_rate', 'github_activity_score',
        'notice_period_days', 'expected_salary_min', 'expected_salary_max',
        'profile_completeness', 'applications_30d', 'interview_completion_rate',
        'timeline_impossible', 'expert_zero_usage_count', 'profile_text',
    }
    defaults = {k: v for k, v in defaults.items() if k in valid_keys}
    return CandidateFeatures(**defaults)

def test_reasoning_returns_nonempty_string():
    f = _make_features()
    r = generate_reasoning(f, rank=1)
    assert isinstance(r, str) and len(r) > 20

def test_reasoning_references_actual_yoe():
    f = _make_features(yoe=7.2)
    r = generate_reasoning(f, rank=1)
    assert "7.2" in r

def test_reasoning_references_actual_title():
    f = _make_features(current_title="Senior ML Engineer")
    r = generate_reasoning(f, rank=1)
    assert "Senior ML Engineer" in r

def test_reasoning_mentions_concern_for_low_ranks():
    f = _make_features(notice_period_days=120, recruiter_response_rate=0.1, rank_=None)
    r_low = generate_reasoning(f, rank=50)
    r_high = generate_reasoning(f, rank=5)
    # Concerns should appear at lower ranks
    assert "concern" in r_low.lower() or "notice" in r_low.lower()

def test_no_two_candidates_produce_identical_reasoning():
    f1 = _make_features(candidate_id="CAND_0000001", yoe=7.0, top_matched_skill="FAISS")
    f2 = _make_features(candidate_id="CAND_0000002", yoe=6.5, top_matched_skill="Python")
    r1 = generate_reasoning(f1, rank=1)
    r2 = generate_reasoning(f2, rank=2)
    assert r1 != r2

def test_reasoning_no_hallucination_no_made_up_company():
    f = _make_features(current_company="Smallco")
    r = generate_reasoning(f, rank=1)
    assert "Google" not in r and "Amazon" not in r
