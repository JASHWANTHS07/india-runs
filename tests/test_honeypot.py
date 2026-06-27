import pytest
from src.features import CandidateFeatures
from src.honeypot import is_honeypot

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
    return CandidateFeatures(**defaults)

def test_clean_candidate_is_not_honeypot():
    assert is_honeypot(_make_features()) is False

def test_impossible_timeline_flags_honeypot():
    assert is_honeypot(_make_features(timeline_impossible=True)) is True

def test_expert_zero_usage_above_threshold_flags_honeypot():
    assert is_honeypot(_make_features(expert_zero_usage_count=4)) is True

def test_expert_zero_usage_at_threshold_not_honeypot():
    # 3 is allowed; 4+ is a flag
    assert is_honeypot(_make_features(expert_zero_usage_count=3)) is False
