"""
Scoring engine for Redrob candidate ranking.

Architecture:
  final_score = relevance_gate * quality_score * availability_modifier
"""

import math
from src.honeypot import is_honeypot


_TITLE_TIER_SCORE = {
    4: 1.00,
    3: 0.60,
    2: 0.25,
    1: 0.08,
    0: 0.02,
}


def title_relevance_score(f):
    base = _TITLE_TIER_SCORE.get(f.title_relevance_tier, 0.02)
    if f.is_cv_only_title:
        if f.career_retrieval_months > 6:
            base *= 0.85
        else:
            base *= 0.30
    return base


def career_depth_score(f):
    depth = min(1.0, f.career_ai_depth_ratio * 1.4)
    retrieval_bonus = min(0.25, f.career_retrieval_months / 48)
    shipped_bonus = min(0.15, f.shipped_count * 0.05)
    product_ai_bonus = 0.10 if f.has_product_ai_career else 0.0
    vec_bonus = 0.05 if f.vector_search_experience else 0.0
    return min(1.0, depth + retrieval_bonus + shipped_bonus + product_ai_bonus + vec_bonus)


def experience_fit_score(technical_yoe):
    if 5.0 <= technical_yoe <= 9.0:
        return math.exp(-0.5 * ((technical_yoe - 7.0) / 1.5) ** 2)
    elif technical_yoe < 5.0:
        return max(0.05, (technical_yoe / 5.0) * 0.3)
    else:
        return max(0.15, 1.0 - (technical_yoe - 9.0) / 8.0)


def skills_quality_score(f):
    base = f.skills_match_score
    if f.jd_skill_assessment_avg >= 0:
        assess_mult = 0.7 + 0.6 * (f.jd_skill_assessment_avg / 100.0)
    else:
        assess_mult = 1.0
    coherence_mult = 0.3 + 0.7 * f.skill_career_coherence
    tier1_bonus = min(0.15, f.jd_tier1_skill_count * 0.03)
    return min(1.0, base * assess_mult * coherence_mult + tier1_bonus)


def behavioral_score(f):
    recency = max(0.0, 1.0 - f.days_since_active / 365.0)
    response = max(0.0, min(1.0, f.recruiter_response_rate))
    speed = max(0.0, 1.0 - f.avg_response_time_hours / 300.0)
    github = (f.github_activity_score / 100.0) if f.github_activity_score >= 0 else 0.35
    market = min(1.0, f.saved_by_recruiters / 15.0) * 0.3 + min(1.0, f.profile_views_30d / 100.0) * 0.2
    profile_q = f.profile_completeness / 100.0
    otw_boost = 1.12 if f.open_to_work else 1.0
    interview_rel = max(0.5, f.interview_completion_rate)
    base = (
        recency * 0.25
        + response * 0.25
        + speed * 0.10
        + github * 0.15
        + market * 0.10
        + profile_q * 0.05
        + interview_rel * 0.10
    )
    return max(0.3, min(1.2, base * otw_boost))


def availability_modifier(f):
    nd = f.notice_period_days
    if nd <= 30:
        notice_factor = 1.00
    elif nd <= 60:
        notice_factor = 0.97
    elif nd <= 90:
        notice_factor = 0.90
    else:
        notice_factor = max(0.65, 1.0 - (nd - 60) / 200.0)
    country = f.country.lower()
    if "india" in country:
        if f.in_preferred_india_city:
            loc_factor = 1.15
        elif f.willing_to_relocate:
            loc_factor = 1.05
        else:
            loc_factor = 0.95
    else:
        loc_factor = 0.50
    return notice_factor * loc_factor


def compute_score(f, semantic_sim):
    if is_honeypot(f):
        return 0.0
    title_score = title_relevance_score(f)
    career_score = career_depth_score(f)
    coherence = f.skill_career_coherence
    relevance = 0.35 * title_score + 0.45 * career_score + 0.20 * coherence
    if relevance < 0.08:
        return relevance * 0.01
    sem_fit = max(0.0, semantic_sim)
    skills_fit = skills_quality_score(f)
    exp_fit = experience_fit_score(f.technical_yoe)
    retrieval_fit = min(1.0, f.career_retrieval_months / 30)
    technical = 0.25 * sem_fit + 0.30 * skills_fit + 0.15 * exp_fit + 0.30 * retrieval_fit
    behav = behavioral_score(f)
    avail = availability_modifier(f)
    base = (0.40 * relevance + 0.40 * technical + 0.20 * behav) * avail
    if f.technical_yoe < 5.0:
        base *= 0.05
    if f.is_consulting_only and f.technical_yoe > 3.0:
        base *= 0.10
    if f.non_tech_title_with_ai_skills and f.ai_ml_months < 12:
        base *= 0.05
    if f.title_relevance_tier == 0 and not f.non_tech_title_with_ai_skills:
        base *= 0.20
    if f.is_cv_only_title and f.career_retrieval_months < 6:
        base *= 0.40
    return max(0.0, min(1.0, base))
