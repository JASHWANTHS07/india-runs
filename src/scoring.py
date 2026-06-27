import math
from src.features import CandidateFeatures
from src.honeypot import is_honeypot


def experience_fit_score(yoe: float) -> float:
    if 5.0 <= yoe <= 9.0:
        return math.exp(-0.5 * ((yoe - 7.0) / 1.5) ** 2)
    elif yoe < 5.0:
        return (yoe / 5.0) * 0.5
    else:
        return max(0.0, 1.0 - (yoe - 9.0) / 6.0)


def career_quality_score(f: CandidateFeatures) -> float:
    score = 0.0
    if f.has_product_company:
        score += 0.30
    if not f.is_consulting_only:
        score += 0.20
    if f.ai_ml_months > 24:
        score += 0.25
    elif f.ai_ml_months > 12:
        score += 0.15
    if f.shipped_count > 0:
        score += 0.15
    if f.vector_search_experience:
        score += 0.10
    return min(1.0, score)


def behavioral_multiplier(f: CandidateFeatures) -> float:
    recency = max(0.0, 1.0 - f.days_since_active / 365.0)
    response = max(0.0, min(1.0, f.recruiter_response_rate))
    github = (f.github_activity_score / 100.0) if f.github_activity_score != -1 else 0.4
    base = recency * 0.40 + response * 0.35 + github * 0.25
    open_boost = 1.15 if f.open_to_work else 1.0
    return max(0.1, min(1.2, base * open_boost))


def availability_multiplier(f: CandidateFeatures) -> float:
    nd = f.notice_period_days
    if nd <= 30:
        notice_factor = 1.00
    elif nd <= 60:
        notice_factor = 0.95
    elif nd <= 90:
        notice_factor = 0.85
    else:
        notice_factor = 0.70

    country = f.country.lower()
    if "india" in country:
        if f.in_preferred_india_city:
            loc_factor = 1.10
        elif f.willing_to_relocate:
            loc_factor = 1.05
        else:
            loc_factor = 0.95
    else:
        loc_factor = 0.70  # no visa sponsorship

    return notice_factor * loc_factor


def compute_score(
    f: CandidateFeatures,
    semantic_sim: float,
    w_semantic: float = 0.35,
    w_career: float = 0.25,
    w_skills: float = 0.25,
    w_experience: float = 0.15,
) -> float:
    if is_honeypot(f):
        return 0.0

    base = (
        w_semantic * semantic_sim
        + w_career * career_quality_score(f)
        + w_skills * f.skills_match_score
        + w_experience * experience_fit_score(f.yoe)
    )

    if f.is_consulting_only and f.yoe > 5.0:
        base *= 0.2
    elif f.ai_ml_months == 0 and "engineer" not in f.current_title.lower() and "scientist" not in f.current_title.lower():
        base *= 0.15

    return base * behavioral_multiplier(f) * availability_multiplier(f)
