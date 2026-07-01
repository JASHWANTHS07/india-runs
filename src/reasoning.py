"""
Generate natural-language reasoning for each ranked candidate.

Lead sentence picks the candidate's most distinctive signal.
JD-connection phrase links at least one signal to a specific JD requirement.
Support section draws from an expanded signal pool.
Concerns appear from rank 3+ (not just tail).
Rank-tier tone: top-10 confident, mid-pack balanced, tail explicitly cautious.
"""

from src.features import NOTABLE_COMPANIES


def generate_reasoning(f, rank, semantic_sim=0.0):
    parts = []
    lead, lead_type = _build_lead(f, rank)
    parts.append(lead)
    second = _build_second_sentence(f, rank, lead_type)
    if second:
        parts.append(second)
    return " ".join(parts)


def _get_domain_string(f):
    if f.career_retrieval_months >= 12:
        return "ranking and retrieval systems"
    if f.ai_ml_months >= 36 and f.has_product_ai_career:
        return "applied AI/ML at product companies"
    if f.ai_ml_months >= 36:
        return "applied AI/ML"
    if f.shipped_count >= 2:
        return "production ML systems"
    return "software engineering"


def _yoe_annotation(f):
    if not hasattr(f, 'technical_yoe') or abs(f.technical_yoe - f.yoe) <= 1.0:
        return ""
    tech_str = str(round(f.technical_yoe, 1))
    yoe_str = str(round(f.yoe, 1))
    if f.technical_yoe > f.yoe:
        return " (" + tech_str + " years verified from career history, " + yoe_str + " stated)"
    return " (" + tech_str + " years non-consulting out of " + yoe_str + " total)"


def _skill_suffix(f):
    if f.top_matched_skill and f.jd_tier1_skill_count >= 2:
        s = "; core competency in " + str(f.top_matched_skill)
        if f.jd_tier1_skill_count >= 4:
            s += " and " + str(f.jd_tier1_skill_count - 1) + " other JD-critical skills"
        return s
    if f.top_matched_skill:
        return "; matched on " + str(f.top_matched_skill)
    return ""


def _build_lead(f, rank):
    notable = getattr(f, 'notable_company', '') or ''
    best_inst = getattr(f, 'best_institution', '') or ''
    best_field = getattr(f, 'best_field', '') or ''
    domain = _get_domain_string(f)
    annot = _yoe_annotation(f)
    suffix = _skill_suffix(f)
    title = str(f.current_title)
    company = str(f.current_company)

    has_notable = bool(notable) and notable.lower() != company.lower()
    has_tier1_edu = f.best_education_tier == 1 and bool(best_inst)
    has_deep_retrieval = f.career_retrieval_months >= 48
    has_multi_shipped = f.shipped_count >= 3
    has_ml_certs = getattr(f, 'ml_cert_count', 0) >= 2
    has_deep_ai = f.ai_ml_months >= 36
    has_retrieval = f.career_retrieval_months >= 12

    # Rank-tier prefix — based on candidate signals, not just rank position
    # Strong candidates ranked low (due to tie-breaking) shouldn't get "borderline"
    is_strong = (f.career_retrieval_months >= 12 and f.has_product_ai_career
                 and f.title_relevance_tier >= 3)
    if rank >= 90 and not is_strong:
        prefix = "Borderline fit: "
    elif rank >= 70 and not is_strong:
        prefix = "Adequate but not strong: "
    else:
        prefix = ""

    # Diverse lead selection — pick the MOST DISTINCTIVE signal per candidate
    # to avoid all top-100 sharing the same "X-year retrieval specialist" template.
    # Use candidate_id hash to break ties when multiple leads qualify.
    cid_hash = sum(ord(c) for c in f.candidate_id) % 7
    retrieval_yrs = str(f.career_retrieval_months // 12) if f.career_retrieval_months >= 12 else "0"
    ai_yrs = str(max(1, f.ai_ml_months // 12)) if f.ai_ml_months >= 12 else str(round(f.yoe, 1))

    # Notable past company — always wins when available (rare: ~3%)
    if has_notable and f.ai_ml_months >= 12:
        lead = prefix + "Former " + notable + " engineer, now " + title + " at " + company + " with " + ai_yrs + "+ years in " + domain + annot + suffix + "."
        return lead, "notable"

    # Tier-1 education — wins when available (relatively rare)
    if has_tier1_edu and (best_field.lower() in _AI_FIELDS or best_field.lower() in _CS_FIELDS or f.education_ai_relevance >= 0.5):
        lead = prefix + title + " at " + company + ", " + best_inst + " " + best_field + " graduate with " + ai_yrs + " years in " + domain + annot + suffix + "."
        return lead, "education"

    # For candidates with deep retrieval — rotate among diverse phrasings
    if f.career_retrieval_months >= 24 and f.has_product_ai_career:
        variant = cid_hash % 5
        if variant == 0:
            lead = prefix + retrieval_yrs + "-year retrieval/ranking specialist, currently " + title + " at " + company + annot + suffix + "."
            return lead, "retrieval_deep"
        elif variant == 1 and f.shipped_count >= 1:
            lead = prefix + title + " at " + company + " with " + str(f.shipped_count) + " shipped search/ranking systems and " + retrieval_yrs + " years in retrieval" + annot + suffix + "."
            return lead, "shipped_retrieval"
        elif variant == 2 and f.vector_search_experience:
            lead = prefix + title + " at " + company + " combining " + retrieval_yrs + " years of retrieval with hands-on vector DB experience" + annot + suffix + "."
            return lead, "vector_retrieval"
        elif variant == 3:
            lead = prefix + title + " at " + company + " with " + ai_yrs + " years of applied AI/ML, including " + retrieval_yrs + " years focused on ranking and retrieval" + annot + suffix + "."
            return lead, "ai_with_retrieval"
        else:
            lead = prefix + title + " at " + company + " with deep production experience in search, ranking, and recommendation systems" + annot + suffix + "."
            return lead, "production_search"

    # Deep retrieval without product AI
    if f.career_retrieval_months >= 48:
        lead = prefix + title + " at " + company + " with " + retrieval_yrs + "+ years building ranking and retrieval systems" + annot + suffix + "."
        return lead, "retrieval"

    # Multiple shipped systems (>=3)
    if has_multi_shipped:
        companies_str = "at product companies" if f.has_product_ai_career else "across multiple roles"
        lead = prefix + title + " at " + company + " with " + str(f.shipped_count) + " production ML deployments " + companies_str + annot + suffix + "."
        return lead, "shipped"

    # ML certifications (>=2)
    if has_ml_certs:
        cert_count = getattr(f, 'ml_cert_count', 0)
        lead = prefix + title + " at " + company + ", holds " + str(cert_count) + " ML/cloud certifications, with " + ai_yrs + " years in " + domain + annot + suffix + "."
        return lead, "certs"

    # Strong AI depth + product
    if has_deep_ai and f.has_product_ai_career:
        lead = prefix + title + " at " + company + " with " + str(f.ai_ml_months // 12) + " years of applied AI/ML at product companies" + annot + suffix + "."
        return lead, "ai_depth"

    # Has retrieval (12+ months)
    if has_retrieval:
        lead = prefix + title + " at " + company + " with " + retrieval_yrs + "+ years in ranking, retrieval, and search systems" + annot + suffix + "."
        return lead, "retrieval"

    # Has shipped
    if f.shipped_count >= 1:
        lead = prefix + title + " at " + company + " with production ML deployment experience" + annot + suffix + "."
        return lead, "shipped"

    # Fallback
    yoe_str = str(round(f.yoe, 1))
    lead = prefix + title + " at " + company + " with " + yoe_str + " years of experience" + annot + suffix + "."
    return lead, "fallback"


_AI_FIELDS = {
    "artificial intelligence", "machine learning", "data science",
    "deep learning", "natural language processing", "computational linguistics",
}

_CS_FIELDS = {
    "computer science", "computer engineering", "software engineering",
    "information technology", "mathematics", "statistics", "physics",
    "electrical engineering", "electronics",
}


def _get_jd_phrase(f):
    """Pick a JD-connection phrase — rotates among applicable phrases for diversity."""
    applicable = []
    if f.has_product_ai_career and f.shipped_count >= 1 and not f.is_consulting_only:
        applicable.append("matches the JD's 'product over research' profile")
    if f.career_retrieval_months >= 24:
        applicable.append("directly fits the JD's production retrieval mandate")
    if f.ai_ml_months >= 48 and f.shipped_count >= 1:
        applicable.append("shows the pre-LLM production ML depth the JD values")
    if f.vector_search_experience and f.career_retrieval_months >= 6:
        applicable.append("vector DB proficiency aligns with the JD's hybrid search requirement")
    if f.has_product_company and not f.is_consulting_only:
        applicable.append("aligns with the JD's product-company preference")
    avg_tenure = getattr(f, 'avg_tenure_months', 0.0)
    if avg_tenure >= 30:
        applicable.append("tenure fits the JD's 3+ year commitment preference")
    if f.github_activity_score >= 60:
        applicable.append("open-source presence provides JD-required external validation")
    assess_ct = getattr(f, 'assessment_jd_count', 0)
    if assess_ct >= 2:
        applicable.append("platform assessment scores provide external skill validation")
    sal_fit = getattr(f, 'salary_fits_role', 0)
    if sal_fit >= 0.5:
        applicable.append("salary expectations align with senior IC compensation at this stage")
    if f.jd_skill_count >= 2:
        applicable.append("skill set overlaps with multiple JD requirements")
    if not applicable:
        return ""
    cid_hash = sum(ord(c) for c in f.candidate_id)
    return applicable[cid_hash % len(applicable)]


def _get_support_signals(f, lead_type):
    """Collect support signal fragments (lowercased, no period)."""
    signals = []
    notable = getattr(f, 'notable_company', '') or ''
    best_inst = getattr(f, 'best_institution', '') or ''
    best_field = getattr(f, 'best_field', '') or ''

    country = str(f.country).lower()
    if "india" in country:
        if f.in_preferred_india_city:
            city_name = str(f.city).split(",")[0].strip().title()
            signals.append(city_name + "-based")
        elif f.willing_to_relocate:
            signals.append("willing to relocate within India")

    if lead_type != "notable" and notable and notable.lower() != str(f.current_company).lower():
        signals.append("previously at " + notable)

    if lead_type != "education" and best_inst and f.best_education_tier <= 2:
        tier_label = {1: "Tier-1", 2: "Tier-2"}.get(f.best_education_tier, "")
        signals.append(tier_label + " " + best_inst + " background")

    ml_certs = getattr(f, 'ml_cert_count', 0)
    if lead_type != "certs" and ml_certs >= 1:
        signals.append(str(ml_certs) + " ML/cloud cert" + ("s" if ml_certs > 1 else ""))

    if f.open_to_work and f.recruiter_response_rate >= 0.7:
        signals.append("strong engagement (" + str(int(f.recruiter_response_rate * 100)) + "% response rate)")
    elif f.open_to_work:
        signals.append("actively seeking roles")
    elif f.recruiter_response_rate >= 0.7:
        signals.append(str(int(f.recruiter_response_rate * 100)) + "% recruiter response rate")

    if f.github_activity_score >= 50:
        signals.append("GitHub score " + str(int(f.github_activity_score)))

    if f.notice_period_days <= 30:
        signals.append("available within 30 days")

    return signals


def _get_concerns(f, rank):
    """Collect concern fragments (lowercased, no period)."""
    concerns = []

    if f.notice_period_days > 90:
        concerns.append("notice period " + str(f.notice_period_days) + "d")
    elif rank >= 3 and f.notice_period_days > 30:
        concerns.append("notice period " + str(f.notice_period_days) + "d (JD prefers sub-30)")
    if f.recruiter_response_rate < 0.25:
        concerns.append("low recruiter responsiveness")
    if f.is_consulting_only:
        concerns.append("consulting-only background")
    if f.days_since_active > 180:
        concerns.append("inactive on platform 6+ months")
    country = str(f.country).lower()
    if "india" not in country:
        concerns.append("based outside India")
    if f.is_cv_only_title:
        concerns.append("CV/robotics specialist rather than NLP/IR")

    if rank >= 10:
        if not f.open_to_work:
            concerns.append("not marked open to opportunities")
        if f.title_relevance_tier <= 1 and f.ai_ml_months > 0:
            concerns.append("non-engineering title")

    if rank >= 40:
        if f.career_retrieval_months < 12 and f.ai_ml_months >= 12:
            concerns.append("limited retrieval-specific experience")
        avg_tenure = getattr(f, 'avg_tenure_months', 0.0)
        if 0 < avg_tenure < 18:
            concerns.append("short tenure pattern")

    if rank >= 80 and not concerns:
        if f.career_retrieval_months < 24:
            concerns.append("weak retrieval depth for this JD")
        if f.shipped_count < 1:
            concerns.append("no confirmed production deployments")

    return concerns


def _build_second_sentence(f, rank, lead_type):
    """Compose JD connection + support + concerns into one sentence."""
    jd = _get_jd_phrase(f)
    support = _get_support_signals(f, lead_type)
    concerns = _get_concerns(f, rank)

    pos_parts = []
    if jd:
        pos_parts.append(jd)
    pos_parts.extend(support[:2])

    neg_parts = concerns[:2]

    if not pos_parts and not neg_parts:
        return ""

    # Build the sentence
    if rank >= 90:
        if neg_parts:
            neg_str = "; ".join(neg_parts)
            if pos_parts:
                pos_str = ", ".join(pos_parts[:2])
                return pos_str[0].upper() + pos_str[1:] + " but ranked as fringe candidate due to " + neg_str + "."
            return "Ranked as fringe candidate: " + neg_str + "."
        pos_str = ", ".join(pos_parts[:2])
        return pos_str[0].upper() + pos_str[1:] + " but overall a borderline fit for this JD."

    if rank >= 70:
        if neg_parts:
            neg_str = "; ".join(neg_parts)
            if pos_parts:
                pos_str = " and ".join(pos_parts[:2]) if len(pos_parts) <= 2 else ", ".join(pos_parts[:2])
                return pos_str[0].upper() + pos_str[1:] + "; however " + neg_str + "."
            return "Some concern: " + neg_str + "."
        pos_str = " and ".join(pos_parts[:2])
        return pos_str[0].upper() + pos_str[1:] + "."

    # Rank 1-69: positive-led, concerns as "but" clause
    if pos_parts and neg_parts:
        pos_str = " and ".join(pos_parts[:2]) if len(pos_parts) <= 2 else ", ".join(pos_parts[:2])
        neg_str = "; ".join(neg_parts[:1])
        return pos_str[0].upper() + pos_str[1:] + "; some concern on " + neg_str + "."

    if pos_parts:
        pos_str = " and ".join(pos_parts[:2]) if len(pos_parts) <= 2 else ", ".join(pos_parts[:3])
        return pos_str[0].upper() + pos_str[1:] + "."

    neg_str = "; ".join(neg_parts)
    return "Some concern: " + neg_str + "."
