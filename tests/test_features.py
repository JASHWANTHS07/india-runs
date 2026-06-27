import pytest
from src.features import extract_features, CandidateFeatures

def make_candidate(overrides: dict = {}) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "ML Engineer",
            "summary": "Built embedding retrieval systems",
            "location": "Pune, India",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "Smallco",
        },
        "career_history": [
            {
                "company": "Smallco",
                "title": "ML Engineer",
                "duration_months": 36,
                "is_current": True,
                "description": "Deployed embedding-based retrieval system using FAISS and sentence-transformers in production. Shipped vector search pipeline serving 1M users.",
            }
        ],
        "education": [{"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS", "tier": "tier_1"}],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "github_activity_score": 75.0,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 40.0},
            "willing_to_relocate": True,
            "interview_completion_rate": 0.9,
            "applications_submitted_30d": 3,
        },
    }
    base.update(overrides)
    return base

def test_basic_extraction():
    f = extract_features(make_candidate())
    assert isinstance(f, CandidateFeatures)
    assert f.candidate_id == "CAND_0000001"
    assert f.yoe == 7.0
    assert f.has_product_company is True
    assert f.is_consulting_only is False

def test_ai_ml_months_detected_from_description():
    f = extract_features(make_candidate())
    assert f.ai_ml_months > 0
    assert f.shipped_count > 0
    assert f.vector_search_experience is True

def test_skills_match_score_nonzero():
    f = extract_features(make_candidate())
    assert 0.0 < f.skills_match_score <= 1.0

def test_consulting_only_detection():
    c = make_candidate()
    c["career_history"][0]["company"] = "Tata Consultancy Services"
    c["profile"]["current_company"] = "Tata Consultancy Services"
    f = extract_features(c)
    assert f.is_consulting_only is True
    assert f.has_product_company is False

def test_behavioral_signals_extracted():
    f = extract_features(make_candidate())
    assert f.open_to_work is True
    assert f.recruiter_response_rate == 0.8
    assert f.github_activity_score == 75.0
    assert f.notice_period_days == 30
    assert f.days_since_active >= 0

def test_honeypot_flags_clean_candidate():
    f = extract_features(make_candidate())
    assert f.timeline_impossible is False
    assert f.expert_zero_usage_count == 0

def test_honeypot_flag_expert_zero_usage():
    c = make_candidate()
    c["skills"] = [
        {"name": skill, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for skill in ["Python", "FAISS", "PyTorch", "Kubernetes", "Go"]
    ]
    f = extract_features(c)
    assert f.expert_zero_usage_count >= 4

def test_profile_text_nonempty():
    f = extract_features(make_candidate())
    assert len(f.profile_text) > 50

def test_education_tier():
    f = extract_features(make_candidate())
    assert f.best_education_tier == 1  # IIT = tier_1

def test_location_preferred_india_city():
    f = extract_features(make_candidate())
    assert f.in_preferred_india_city is True  # Pune
