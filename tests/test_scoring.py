import pytest
import math
from src.features import CandidateFeatures
from src.scoring import (
    experience_fit_score, career_depth_score,
    behavioral_score, availability_modifier, compute_score,
)

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

# --- career_depth_score ---
def test_career_depth_full_marks(make_features):
    f = make_features(
        has_product_company=True, is_consulting_only=False,
        ai_ml_months=36, shipped_count=2, vector_search_experience=True
    )
    score = career_depth_score(f)
    assert score == pytest.approx(1.0, abs=0.01)

def test_career_depth_consulting_only_zero_ai(make_features):
    f = make_features(
        has_product_company=False, is_consulting_only=True,
        ai_ml_months=0, shipped_count=0, vector_search_experience=False,
        career_ai_depth_ratio=0.0, career_retrieval_months=0,
        has_product_ai_career=False
    )
    score = career_depth_score(f)
    assert score < 0.3

# --- behavioral_score ---
def test_behavioral_open_to_work_boost(make_features):
    f_open = make_features(open_to_work=True, days_since_active=7,
                             recruiter_response_rate=0.8, github_activity_score=70.0)
    f_closed = make_features(open_to_work=False, days_since_active=7,
                               recruiter_response_rate=0.8, github_activity_score=70.0)
    assert behavioral_score(f_open) > behavioral_score(f_closed)

def test_behavioral_inactive_candidate_low_score(make_features):
    f = make_features(days_since_active=400, recruiter_response_rate=0.05,
                       github_activity_score=-1, open_to_work=False,
                       avg_response_time_hours=200, saved_by_recruiters=0,
                       profile_views_30d=0, profile_completeness=50,
                       interview_completion_rate=0.5)
    assert behavioral_score(f) <= 0.35

def test_behavioral_score_clamped(make_features):
    f = make_features(days_since_active=0, recruiter_response_rate=1.0,
                       github_activity_score=100.0, open_to_work=True)
    m = behavioral_score(f)
    assert 0.3 <= m <= 1.2

# --- availability_modifier ---
def test_availability_preferred_city_short_notice(make_features):
    f = make_features(notice_period_days=30, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    assert availability_modifier(f) > 1.0

def test_availability_overseas_penalty(make_features):
    f = make_features(notice_period_days=30, country="united states",
                       in_preferred_india_city=False, willing_to_relocate=False)
    assert availability_modifier(f) < 0.8

def test_availability_long_notice_penalty(make_features):
    f = make_features(notice_period_days=120, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    assert availability_modifier(f) < availability_modifier(
        make_features(notice_period_days=30, country="india",
                       in_preferred_india_city=True, willing_to_relocate=True)
    )

# --- compute_score ---
def test_honeypot_scores_zero(make_features):
    f = make_features(timeline_impossible=True)
    assert compute_score(f, semantic_sim=0.9) == 0.0

def test_consulting_only_heavily_penalized(make_features):
    f_prod = make_features(is_consulting_only=False, has_product_company=True, yoe=5.5)
    f_cons = make_features(is_consulting_only=True, has_product_company=False, yoe=5.5)
    assert compute_score(f_prod, 0.7) > compute_score(f_cons, 0.7) * 3

def test_great_candidate_scores_high(make_features):
    f = make_features()  # all defaults are ideal
    score = compute_score(f, semantic_sim=0.85)
    assert score > 0.35
