import pytest
from src.features import CandidateFeatures


@pytest.fixture
def make_features():
    def _make(**overrides):
        defaults = dict(
            candidate_id="CAND_0000001", yoe=7.0, current_title="ML Engineer",
            current_company="Smallco",
            title_relevance_tier=4, is_cv_only_title=False,
            has_product_company=True, is_consulting_only=False,
            ai_ml_months=36, shipped_count=2, vector_search_experience=True,
            career_retrieval_months=24, career_ai_depth_ratio=0.5,
            has_product_ai_career=True,
            top_matched_skill="FAISS", skills_match_score=0.6,
            jd_skill_count=5, jd_tier1_skill_count=3,
            skill_career_coherence=0.6, non_tech_title_with_ai_skills=False,
            jd_skill_assessment_avg=60.0,
            best_education_tier=2, country="india", city="pune, india",
            in_preferred_india_city=True, willing_to_relocate=True,
            open_to_work=True, last_active_date="2026-06-20",
            days_since_active=7, recruiter_response_rate=0.8,
            github_activity_score=70.0, notice_period_days=30,
            expected_salary_min=25.0, expected_salary_max=40.0,
            profile_completeness=85.0, applications_30d=3,
            interview_completion_rate=0.9, saved_by_recruiters=10,
            profile_views_30d=50, avg_response_time_hours=24.0,
            verified_count=2, endorsements_total=100,
            connection_count=200, offer_acceptance_rate=0.7,
            consulting_months=0, product_months=84,
            technical_yoe=7.0, timeline_impossible=False,
            expert_zero_usage_count=0, has_cs_degree=True,
            highest_degree_level=3, education_ai_relevance=0.5,
            education_recency=5, cert_count=2, ml_cert_count=1,
            cert_recency=2, ai_title_count=2, title_progression=1,
            avg_tenure_months=36.0, num_roles=3,
            max_company_size_ord=6, current_company_size_ord=5,
            total_skill_count=12, avg_skill_proficiency=0.65,
            endorsed_skill_ratio=0.7, skill_keyword_density=0.4,
            work_mode_match=1.0, search_appearance_30d=100,
            salary_range_width=15.0, platform_tenure_days=365,
            headline_has_ai_keywords=True,
            headline_has_generic_filler=False,
            salary_inverted=False, salary_fits_role=0.5,
            assessment_count=3, assessment_jd_count=2,
            assessment_proficiency_gap=0.5,
            market_demand_score=0.5,
            summary_is_template=False, summary_ai_keyword_count=5,
            career_desc_title_mismatch_count=0,
            career_production_keyword_density=2.0,
            notable_company="", best_institution="",
            best_field="", profile_text="...",
        )
        defaults.update(overrides)
        return CandidateFeatures(**defaults)
    return _make
