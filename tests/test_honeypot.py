import pytest
from src.features import CandidateFeatures
from src.honeypot import is_honeypot

def test_clean_candidate_is_not_honeypot(make_features):
    assert is_honeypot(make_features()) is False

def test_impossible_timeline_flags_honeypot(make_features):
    assert is_honeypot(make_features(timeline_impossible=True)) is True

def test_expert_zero_usage_above_threshold_flags_honeypot(make_features):
    assert is_honeypot(make_features(expert_zero_usage_count=4)) is True

def test_expert_zero_usage_at_threshold_not_honeypot(make_features):
    # 3 is allowed; 4+ is a flag
    assert is_honeypot(make_features(expert_zero_usage_count=3)) is False

# --- New honeypot conditions ---

def test_salary_inverted_flags_honeypot(make_features):
    assert is_honeypot(make_features(salary_inverted=True)) is True

def test_assessment_proficiency_gap_flags_honeypot(make_features):
    assert is_honeypot(make_features(
        assessment_proficiency_gap=2.5, assessment_count=3
    )) is True

def test_template_summary_with_mismatch_flags_honeypot(make_features):
    assert is_honeypot(make_features(
        summary_is_template=True, career_desc_title_mismatch_count=2
    )) is True

def test_endorsement_anomaly_flags_honeypot(make_features):
    assert is_honeypot(make_features(
        endorsements_total=100, connection_count=5
    )) is True

def test_ghost_profile_flags_honeypot(make_features):
    assert is_honeypot(make_features(
        profile_completeness=90, days_since_active=200,
        applications_30d=0, profile_views_30d=0
    )) is True
