import pytest
import math
from src.features import CandidateFeatures
from src.scoring import (
    experience_fit_score, career_quality_score,
    behavioral_multiplier, availability_multiplier, compute_score,
)

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

# --- experience_fit_score ---
def test_experience_fit_peak_at_7():
    score_7 = experience_fit_score(7.0)
    score_6 = experience_fit_score(6.0)
    score_8 = experience_fit_score(8.0)
    assert score_7 > score_6
    assert score_7 > score_8
    assert score_7 == pytest.approx(1.0, abs=0.01)

def test_experience_fit_junior_penalty():
    assert experience_fit_score(3.0) < 0.5

def test_experience_fit_over_qualified_penalty():
    score_9 = experience_fit_score(9.0)
    score_15 = experience_fit_score(15.0)
    assert score_15 < score_9

# --- career_quality_score ---
def test_career_quality_full_marks():
    f = _make_features(
        has_product_company=True, is_consulting_only=False,
        ai_ml_months=36, shipped_count=2, vector_search_experience=True
    )
    score = career_quality_score(f)
    assert score == pytest.approx(1.0, abs=0.01)

def test_career_quality_consulting_only_zero_ai():
    f = _make_features(
        has_product_company=False, is_consulting_only=True,
        ai_ml_months=0, shipped_count=0, vector_search_experience=False
    )
    score = career_quality_score(f)
    assert score < 0.3

# --- behavioral_multiplier ---
def test_behavioral_open_to_work_boost():
    f_open = _make_features(open_to_work=True, days_since_active=7,
                             recruiter_response_rate=0.8, github_activity_score=70.0)
    f_closed = _make_features(open_to_work=False, days_since_active=7,
                               recruiter_response_rate=0.8, github_activity_score=70.0)
    assert behavioral_multiplier(f_open) > behavioral_multiplier(f_closed)

def test_behavioral_inactive_candidate_low_multiplier():
    f = _make_features(days_since_active=400, recruiter_response_rate=0.05,
                       github_activity_score=-1, open_to_work=False)
    assert behavioral_multiplier(f) < 0.3

def test_behavioral_multiplier_clamped():
    f = _make_features(days_since_active=0, recruiter_response_rate=1.0,
                       github_activity_score=100.0, open_to_work=True)
    m = behavioral_multiplier(f)
    assert 0.1 <= m <= 1.2

# --- availability_multiplier ---
def test_availability_preferred_city_short_notice():
    f = _make_features(notice_period_days=30, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    assert availability_multiplier(f) > 1.0

def test_availability_overseas_penalty():
    f = _make_features(notice_period_days=30, country="united states",
                       in_preferred_india_city=False, willing_to_relocate=False)
    assert availability_multiplier(f) < 0.8

def test_availability_long_notice_penalty():
    f = _make_features(notice_period_days=120, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    assert availability_multiplier(f) < availability_multiplier(
        _make_features(notice_period_days=30, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    )

# --- compute_score ---
def test_honeypot_scores_zero():
    f = _make_features(timeline_impossible=True)
    assert compute_score(f, semantic_sim=0.9) == 0.0

def test_consulting_only_heavily_penalized():
    f_prod = _make_features(is_consulting_only=False, has_product_company=True, yoe=5.5)
    f_cons = _make_features(is_consulting_only=True, has_product_company=False, yoe=5.5)
    assert compute_score(f_prod, 0.7) > compute_score(f_cons, 0.7) * 3

def test_great_candidate_scores_high():
    f = _make_features()  # all defaults are ideal
    score = compute_score(f, semantic_sim=0.85)
    assert score > 0.5
