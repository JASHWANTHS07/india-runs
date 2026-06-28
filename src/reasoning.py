"""
Generate natural-language reasoning for each ranked candidate.
"""


def generate_reasoning(f, rank, semantic_sim=0.0):
    parts = []
    parts.append(_build_lead(f))
    support = _build_support(f, rank)
    if support:
        parts.append(support)
    if rank > 15:
        concerns = _build_concerns(f, rank)
        if concerns:
            parts.append(concerns)
    return " ".join(parts)


def _build_lead(f):
    has_retrieval = f.career_retrieval_months >= 12
    has_deep_ai = f.ai_ml_months >= 36
    has_shipped = f.shipped_count >= 2
    has_product = f.has_product_ai_career
    yoe_str = str(round(f.yoe, 1))
    if has_retrieval and has_product:
        cd = "with " + str(f.career_retrieval_months // 12) + "+ years building ranking and retrieval systems at product companies"
    elif has_retrieval:
        cd = "with " + str(f.career_retrieval_months // 12) + "+ years in ranking, retrieval, and search systems"
    elif has_deep_ai and has_product:
        cd = "with " + str(f.ai_ml_months // 12) + " years of applied AI/ML at product companies"
    elif has_deep_ai:
        cd = "with " + str(f.ai_ml_months // 12) + " years of applied AI/ML experience"
    elif has_shipped:
        cd = "with multiple production ML deployments"
    else:
        cd = "with " + yoe_str + " years of experience"
    lead = str(f.current_title) + " at " + str(f.current_company) + " " + cd
    if f.top_matched_skill and f.jd_tier1_skill_count >= 2:
        lead += "; core competency in " + str(f.top_matched_skill)
        if f.jd_tier1_skill_count >= 4:
            lead += " and " + str(f.jd_tier1_skill_count - 1) + " other JD-critical skills"
    elif f.top_matched_skill:
        lead += "; matched on " + str(f.top_matched_skill)
    return lead + "."


def _build_support(f, rank):
    signals = []
    country = str(f.country).lower()
    if "india" in country:
        if f.in_preferred_india_city:
            city_name = str(f.city).split(",")[0].strip()
            signals.append(city_name + "-based (preferred location)")
        elif f.willing_to_relocate:
            signals.append("willing to relocate within India")
    if f.open_to_work and f.recruiter_response_rate >= 0.7:
        rr = str(int(f.recruiter_response_rate * 100))
        signals.append("actively seeking roles with " + rr + "% recruiter response rate")
    elif f.open_to_work:
        signals.append("actively open to new roles")
    elif f.recruiter_response_rate >= 0.7:
        rr = str(int(f.recruiter_response_rate * 100))
        signals.append("strong recruiter engagement (" + rr + "% response rate)")
    if f.github_activity_score >= 50:
        signals.append("active open-source contributor (GitHub score " + str(int(f.github_activity_score)) + ")")
    if f.notice_period_days <= 30:
        signals.append("available within 30 days")
    if f.jd_skill_assessment_avg >= 60:
        signals.append("strong platform assessment scores (avg " + str(int(f.jd_skill_assessment_avg)) + "/100)")
    if rank <= 10 and f.best_education_tier <= 2:
        tn = {1: "Tier-1", 2: "Tier-2"}.get(f.best_education_tier, "")
        signals.append(tn + " institution background")
    if not signals:
        return ""
    sel = signals[:3]
    first = sel[0][0].upper() + sel[0][1:]
    if len(sel) == 1:
        return first + "."
    elif len(sel) == 2:
        return first + ", " + sel[1] + "."
    else:
        return first + ", " + sel[1] + ", and " + sel[2] + "."


def _build_concerns(f, rank):
    concerns = []
    if f.notice_period_days > 90:
        concerns.append("extended notice period (" + str(f.notice_period_days) + "d)")
    if f.recruiter_response_rate < 0.25:
        concerns.append("low recruiter responsiveness")
    if not f.open_to_work and rank > 30:
        concerns.append("not currently marked open to opportunities")
    if f.is_consulting_only:
        concerns.append("consulting-only career background")
    if f.days_since_active > 180:
        concerns.append("inactive on platform for 6+ months")
    country = str(f.country).lower()
    if "india" not in country:
        concerns.append("based outside India (" + str(f.country) + ")")
    if f.is_cv_only_title:
        concerns.append("primary expertise in computer vision rather than NLP/IR")
    if f.title_relevance_tier <= 1 and f.ai_ml_months > 0:
        concerns.append("non-engineering title despite some AI career exposure")
    if not concerns:
        return ""
    sel = concerns[:2]
    return "Some concern: " + "; ".join(sel) + "."
