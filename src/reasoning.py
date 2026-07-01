"""
Generate natural-language reasoning for each ranked candidate.

Three-part narrative structure per candidate:
  1. Identity — who they are + most distinctive signal
  2. Technical evidence — specific skills, depth, JD alignment
  3. Context — supporting signals and/or honest concerns

Each part uses multiple template variants rotated by candidate_id hash
to ensure no two candidates produce structurally identical reasoning.
"""

from src.features import NOTABLE_COMPANIES


def generate_reasoning(f, rank, semantic_sim=0.0):
    h1 = _hash(f.candidate_id, 0)
    h2 = _hash(f.candidate_id, 17)
    h3 = _hash(f.candidate_id, 31)

    parts = [_build_identity(f, rank, h1)]

    tech = _build_technical(f, h2)
    if tech:
        parts.append(tech)

    ctx = _build_context(f, rank, h3)
    if ctx:
        parts.append(ctx)

    return " ".join(parts)


def _hash(cid, salt=0):
    return sum((ord(c) + salt) * (i + 1) for i, c in enumerate(cid))


# ---------------------------------------------------------------------------
# Part 1: Identity — who + most distinctive signal
# ---------------------------------------------------------------------------

_AI_FIELDS = {
    "artificial intelligence", "machine learning", "data science",
    "deep learning", "natural language processing", "computational linguistics",
}

_CS_FIELDS = {
    "computer science", "computer engineering", "software engineering",
    "information technology", "mathematics", "statistics", "physics",
    "electrical engineering", "electronics",
}


def _domain(f):
    if f.career_retrieval_months >= 12:
        return "ranking and retrieval"
    if f.ai_ml_months >= 36 and f.has_product_ai_career:
        return "applied AI/ML at product companies"
    if f.ai_ml_months >= 24:
        return "applied AI/ML"
    if f.shipped_count >= 2:
        return "production ML"
    return "software engineering"


def _rank_prefix(f, rank):
    is_strong = (f.career_retrieval_months >= 12 and f.has_product_ai_career
                 and f.title_relevance_tier >= 3)
    if rank >= 90 and not is_strong:
        return "Borderline fit: "
    if rank >= 70 and not is_strong:
        return "Adequate but not strong: "
    return ""


def _build_identity(f, rank, h):
    pfx = _rank_prefix(f, rank)
    title = _safe_str(f.current_title) or "Engineer"
    company = _safe_str(f.current_company) or "current employer"
    notable = _safe_str(getattr(f, 'notable_company', ''))
    inst = _safe_str(getattr(f, 'best_institution', ''))
    field = _safe_str(getattr(f, 'best_field', ''))
    domain = _domain(f)

    r_mo = _safe_int(f, 'career_retrieval_months')
    ai_mo = _safe_int(f, 'ai_ml_months')
    r_yrs = r_mo // 12
    ai_yrs = max(1, ai_mo // 12) if ai_mo >= 12 else round(_safe_float(f, 'yoe', 1.0), 1)
    yoe = round(_safe_float(f, 'yoe', 1.0), 1)
    career_count = _safe_int(f, 'num_roles', _safe_int(f, 'career_count', 1))

    has_notable = bool(notable) and notable.lower() != company.lower()
    edu_tier = _safe_int(f, 'best_education_tier', 99)
    edu_rel = _safe_float(f, 'education_ai_relevance')
    has_edu = edu_tier == 1 and bool(inst) and (
        field.lower() in _AI_FIELDS or field.lower() in _CS_FIELDS
        or edu_rel >= 0.5)

    # Notable company leads
    if has_notable and f.ai_ml_months >= 12:
        templates = [
            pfx + "Former " + notable + " engineer with " + str(ai_yrs) + " years in " + domain + ", now " + title + " at " + company + ".",
            pfx + title + " at " + company + " who previously built ML systems at " + notable + ", bringing " + str(ai_yrs) + " years of " + domain + " experience.",
            pfx + "Brings " + notable + " pedigree to " + company + " as " + title + ", with " + str(ai_yrs) + " years spanning " + domain + ".",
        ]
        return templates[h % len(templates)]

    # Tier-1 education leads
    if has_edu:
        templates = [
            pfx + inst + " " + field + " graduate, now " + title + " at " + company + " with " + str(ai_yrs) + " years in " + domain + ".",
            pfx + title + " at " + company + " with an " + inst + " " + field + " background and " + str(ai_yrs) + " years of " + domain + " experience.",
            pfx + "Trained in " + field + " at " + inst + ", now applying " + str(ai_yrs) + " years of " + domain + " expertise as " + title + " at " + company + ".",
        ]
        return templates[h % len(templates)]

    # Deep retrieval (24+ months) at product companies
    if r_yrs >= 2 and f.has_product_ai_career:
        templates = [
            pfx + str(r_yrs) + "-year retrieval specialist, currently " + title + " at " + company + ".",
            pfx + title + " at " + company + " who has spent " + str(r_yrs) + " years building search and ranking systems at product companies.",
            pfx + "Has built ranking systems for " + str(r_yrs) + " years across product companies, currently " + title + " at " + company + ".",
            pfx + title + " at " + company + " with " + str(ai_yrs) + " years of applied ML, " + str(r_yrs) + " of which focused on retrieval and ranking.",
            pfx + "Currently " + title + " at " + company + ", with " + str(r_yrs) + " years focused specifically on search, ranking, and retrieval systems.",
        ]
        return templates[h % len(templates)]

    # Multiple shipped systems
    if f.shipped_count >= 3:
        ctx = "at product companies" if f.has_product_ai_career else "across " + str(career_count) + " roles"
        templates = [
            pfx + title + " at " + company + " who has shipped " + str(f.shipped_count) + " production ML systems " + ctx + ".",
            pfx + "Has " + str(f.shipped_count) + " production ML deployments " + ctx + ", currently " + title + " at " + company + ".",
        ]
        return templates[h % len(templates)]

    # Strong AI depth
    if f.ai_ml_months >= 36 and f.has_product_ai_career:
        templates = [
            pfx + title + " at " + company + " with " + str(ai_yrs) + " years of production AI/ML at product companies.",
            pfx + str(ai_yrs) + "-year ML practitioner across " + str(career_count) + " roles, currently " + title + " at " + company + ".",
            pfx + "Seasoned AI/ML engineer with " + str(ai_yrs) + " years at product companies, currently " + title + " at " + company + ".",
        ]
        return templates[h % len(templates)]

    # Has retrieval (12+ months)
    if r_yrs >= 1:
        templates = [
            pfx + title + " at " + company + " with " + str(r_yrs) + "+ years in ranking and retrieval systems.",
            pfx + title + " at " + company + " whose career includes " + str(r_yrs) + " years of retrieval-specific work.",
        ]
        return templates[h % len(templates)]

    # Has shipped (1-2)
    if f.shipped_count >= 1:
        return pfx + title + " at " + company + " with " + str(f.shipped_count) + " production ML deployment" + ("s" if f.shipped_count > 1 else "") + " and " + str(yoe) + " years of experience."

    # Fallback
    templates = [
        pfx + title + " at " + company + " with " + str(yoe) + " years of experience in " + domain + ".",
        pfx + str(yoe) + "-year " + domain + " professional, currently " + title + " at " + company + ".",
    ]
    return templates[h % len(templates)]


# ---------------------------------------------------------------------------
# Part 2: Technical evidence — skills, depth, JD alignment
# ---------------------------------------------------------------------------

def _safe_str(val):
    """Safely convert a field value to string, returning '' for None/NaN."""
    if val is None:
        return ""
    s = str(val)
    if s.lower() in ("none", "nan", ""):
        return ""
    return s


def _safe_int(f, attr, default=0):
    """Safely get an int attribute."""
    val = getattr(f, attr, default)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(f, attr, default=0.0):
    """Safely get a float attribute."""
    val = getattr(f, attr, default)
    if val is None:
        return default
    try:
        v = float(val)
        return default if v != v else v  # NaN check
    except (ValueError, TypeError):
        return default


def _build_technical(f, h):
    parts = []
    top_skill = _safe_str(f.top_matched_skill)
    t1 = _safe_int(f, 'jd_tier1_skill_count')
    jd_ct = _safe_int(f, 'jd_skill_count')
    r_mo = _safe_int(f, 'career_retrieval_months')
    ai_mo = _safe_int(f, 'ai_ml_months')
    shipped = _safe_int(f, 'shipped_count')
    vec = getattr(f, 'vector_search_experience', False)
    coherence = _safe_float(f, 'skill_career_coherence')

    # Skill evidence
    if top_skill and t1 >= 3:
        skill_phrases = [
            "Core competency in " + top_skill + " with " + str(t1) + " JD-critical skills matched",
            "Primary strength in " + top_skill + ", plus " + str(t1 - 1) + " other core JD skills",
            "Skill profile centers on " + top_skill + " and " + str(t1 - 1) + " other JD-priority areas",
            top_skill + " expertise backed by " + str(t1) + " JD-aligned technical skills",
        ]
        parts.append(skill_phrases[h % len(skill_phrases)])
    elif top_skill and t1 >= 1:
        extra = " and " + str(t1 - 1) + " other JD skill" + ("s" if t1 > 2 else "") if t1 >= 2 else ""
        parts.append("matched on " + top_skill + extra)
    elif top_skill:
        parts.append("some skill overlap via " + top_skill)
    elif jd_ct >= 2:
        parts.append(str(jd_ct) + " JD-relevant skills identified")

    # Depth evidence
    if r_mo >= 24 and shipped >= 2:
        depth_phrases = [
            str(r_mo) + " months of retrieval work with " + str(shipped) + " shipped systems",
            str(shipped) + " production deployments across " + str(r_mo // 12) + " years of retrieval-focused work",
        ]
        parts.append(depth_phrases[h % len(depth_phrases)])
    elif r_mo >= 12:
        r_yrs = r_mo // 12
        parts.append(str(r_yrs) + " year" + ("s" if r_yrs != 1 else "") + " in retrieval/ranking roles")
    elif ai_mo >= 24 and shipped >= 1:
        yrs = ai_mo // 12
        parts.append(str(yrs) + " year" + ("s" if yrs != 1 else "") + " in AI/ML with " + str(shipped) + " production deployment" + ("s" if shipped > 1 else ""))
    elif ai_mo >= 12:
        yrs = ai_mo // 12
        parts.append(str(yrs) + " year" + ("s" if yrs != 1 else "") + " in AI/ML roles")

    # Vector DB
    if vec and "vector" not in " ".join(parts).lower():
        parts.append("hands-on vector DB experience")

    # Coherence flag (only if notably low and has enough skills to judge)
    if coherence < 0.2 and jd_ct >= 3:
        parts.append("though skill-career coherence is low (" + str(round(coherence, 2)) + ")")

    if not parts:
        return ""

    # Join — use semicolons for clarity when multiple parts
    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        connectors = ["; ", " — ", " and ", ", plus "]
        joined = parts[0] + connectors[h % len(connectors)] + parts[1]
    else:
        joined = "; ".join(parts)

    # Sentence wrapper
    wrappers = [
        joined + ".",
        "Technical profile: " + joined + ".",
    ]
    return wrappers[h % len(wrappers)]


# ---------------------------------------------------------------------------
# Part 3: Context — JD connection, support, concerns
# ---------------------------------------------------------------------------

def _build_context(f, rank, h):
    positives = []
    concerns = []

    # JD connection phrases
    jd = _jd_phrase(f, h)
    if jd:
        positives.append(jd)

    # Location
    country = _safe_str(getattr(f, 'country', '')).lower()
    if "india" in country and getattr(f, 'in_preferred_india_city', False):
        city = _safe_str(getattr(f, 'city', '')).split(",")[0].strip().title()
        if city:
            positives.append(city + "-based")
    elif "india" in country and getattr(f, 'willing_to_relocate', False):
        positives.append("India-based, willing to relocate")

    # Engagement
    rr = _safe_float(f, 'recruiter_response_rate')
    otw = getattr(f, 'open_to_work', False)
    if otw and rr >= 0.7:
        positives.append(str(int(rr * 100)) + "% recruiter response rate and actively looking")
    elif rr >= 0.7:
        positives.append(str(int(rr * 100)) + "% recruiter response rate")
    elif otw:
        positives.append("actively seeking new roles")

    # GitHub
    gh = _safe_float(f, 'github_activity_score', -1)
    if gh >= 60:
        positives.append("GitHub activity score " + str(int(gh)) + " (external validation)")

    # Availability
    if _safe_int(f, 'notice_period_days') <= 30:
        positives.append("available within 30 days")

    # Education (if not in lead)
    inst = _safe_str(getattr(f, 'best_institution', ''))
    edu_tier = _safe_int(f, 'best_education_tier', 99)
    if inst and edu_tier <= 2:
        tier = {1: "Tier-1", 2: "Tier-2"}.get(edu_tier, "")
        positives.append(tier + " " + inst + " background")

    # Notable company (if not in lead)
    notable = _safe_str(getattr(f, 'notable_company', ''))
    if notable and notable.lower() != _safe_str(f.current_company).lower():
        positives.append("previously at " + notable)

    # Certs
    ml_certs = _safe_int(f, 'ml_cert_count')
    if ml_certs >= 2:
        positives.append(str(ml_certs) + " ML/cloud certifications")

    # --- Concerns ---
    notice = _safe_int(f, 'notice_period_days')
    response_rate = _safe_float(f, 'recruiter_response_rate')
    days_inactive = _safe_int(f, 'days_since_active')

    if notice > 90:
        concerns.append(str(notice) + "-day notice period")
    elif rank >= 5 and notice > 30:
        concerns.append(str(notice) + "d notice (JD prefers sub-30)")

    if response_rate < 0.20:
        concerns.append("low platform responsiveness (" + str(int(response_rate * 100)) + "%)")

    if getattr(f, 'is_consulting_only', False):
        concerns.append("consulting-only career background")

    if days_inactive > 180:
        concerns.append("inactive on platform for " + str(days_inactive // 30) + " months")

    if "india" not in country:
        concerns.append("based outside India")

    if getattr(f, 'is_cv_only_title', False):
        concerns.append("CV/robotics focus rather than NLP/IR")

    if rank >= 10 and not getattr(f, 'open_to_work', False):
        concerns.append("not marked open to work")

    r_mo = _safe_int(f, 'career_retrieval_months')
    ai_mo = _safe_int(f, 'ai_ml_months')
    shipped = _safe_int(f, 'shipped_count')

    if rank >= 40:
        if r_mo < 12 and ai_mo >= 12:
            concerns.append("limited retrieval-specific depth")
        avg_tenure = _safe_float(f, 'avg_tenure_months')
        if 0 < avg_tenure < 18:
            concerns.append("avg tenure " + str(int(avg_tenure)) + " months (short)")

    if rank >= 80 and not concerns:
        if r_mo < 24:
            concerns.append("retrieval depth below top-tier for this JD")
        if shipped < 1:
            concerns.append("no confirmed production deployments")

    # --- Assemble ---

    # Deduplicate positives against what's already in lead/technical
    # (keep first 3 positives max, first 2 concerns max)
    pos = positives[:3]
    neg = concerns[:2]

    if not pos and not neg:
        return ""

    # Varied assembly based on rank tier and hash
    if rank <= 5:
        return _assemble_top(pos, neg, h)
    if rank <= 30:
        return _assemble_strong(pos, neg, h)
    if rank <= 69:
        return _assemble_mid(pos, neg, h)
    if rank <= 89:
        return _assemble_lower(pos, neg, h)
    return _assemble_tail(pos, neg, h)


def _assemble_top(pos, neg, h):
    if pos:
        joined = ", ".join(pos[:3])
        starters = [
            joined[0].upper() + joined[1:] + ".",
            "Strong signals: " + joined + ".",
        ]
        s = starters[h % len(starters)]
        if neg:
            s = s[:-1] + "; minor note: " + neg[0] + "."
        return s
    return ""


def _assemble_strong(pos, neg, h):
    if pos and neg:
        p = ", ".join(pos[:2])
        templates = [
            p[0].upper() + p[1:] + "; some concern on " + neg[0] + ".",
            p[0].upper() + p[1:] + " — though " + neg[0] + ".",
            p[0].upper() + p[1:] + ". Note: " + neg[0] + ".",
        ]
        return templates[h % len(templates)]
    if pos:
        p = " and ".join(pos[:2])
        return p[0].upper() + p[1:] + "."
    return "Some concern: " + "; ".join(neg[:2]) + "."


def _assemble_mid(pos, neg, h):
    if pos and neg:
        p = ", ".join(pos[:2])
        n = "; ".join(neg[:2])
        templates = [
            p[0].upper() + p[1:] + ", but " + n + ".",
            p[0].upper() + p[1:] + "; however " + n + ".",
            "Positives: " + p + ". Concern: " + n + ".",
        ]
        return templates[h % len(templates)]
    if pos:
        p = " and ".join(pos[:2])
        return p[0].upper() + p[1:] + "."
    if neg:
        return "Concern: " + "; ".join(neg) + "."
    return ""


def _assemble_lower(pos, neg, h):
    if pos and neg:
        p = pos[0]
        n = "; ".join(neg[:2])
        templates = [
            p[0].upper() + p[1:] + ", but weighed down by " + n + ".",
            p[0].upper() + p[1:] + "; ranked lower due to " + n + ".",
        ]
        return templates[h % len(templates)]
    if neg:
        n = "; ".join(neg[:2])
        templates = [
            "Ranked in lower tier due to " + n + ".",
            "Main gaps: " + n + ".",
        ]
        return templates[h % len(templates)]
    if pos:
        return pos[0][0].upper() + pos[0][1:] + "."
    return ""


def _assemble_tail(pos, neg, h):
    if neg:
        n = "; ".join(neg[:2])
        if pos:
            p = pos[0]
            templates = [
                p[0].upper() + p[1:] + " but fringe candidate due to " + n + ".",
                "Despite " + p + ", ranked at tail: " + n + ".",
            ]
            return templates[h % len(templates)]
        templates = [
            "Fringe candidate: " + n + ".",
            "At the edge of top 100: " + n + ".",
        ]
        return templates[h % len(templates)]
    if pos:
        p = ", ".join(pos[:2])
        return p[0].upper() + p[1:] + " — borderline fit for this specific JD."
    return ""


def _jd_phrase(f, h):
    """Pick a JD-connection phrase — rotates among applicable ones."""
    applicable = []
    shipped = _safe_int(f, 'shipped_count')
    r_mo = _safe_int(f, 'career_retrieval_months')
    ai_mo = _safe_int(f, 'ai_ml_months')

    if getattr(f, 'has_product_ai_career', False) and shipped >= 1 and not getattr(f, 'is_consulting_only', False):
        applicable.append("fits the JD's 'product over research' requirement")
    if r_mo >= 24:
        applicable.append("directly matches the JD's retrieval systems mandate")
    if ai_mo >= 48 and shipped >= 1:
        applicable.append("pre-LLM production ML depth that the JD specifically values")
    if getattr(f, 'vector_search_experience', False) and r_mo >= 6:
        applicable.append("vector DB proficiency aligns with JD's hybrid search needs")
    if getattr(f, 'has_product_company', False) and not getattr(f, 'is_consulting_only', False):
        applicable.append("product-company track record matches JD preference")
    avg_tenure = _safe_float(f, 'avg_tenure_months')
    if avg_tenure >= 30:
        applicable.append("strong tenure pattern (" + str(int(avg_tenure)) + " months avg)")
    gh = _safe_float(f, 'github_activity_score', -1)
    if gh >= 60:
        applicable.append("open-source activity provides the external validation JD requires")
    assess_ct = _safe_int(f, 'assessment_jd_count')
    if assess_ct >= 2:
        applicable.append("platform assessments back up claimed proficiency")
    sal_fit = _safe_float(f, 'salary_fits_role')
    if sal_fit >= 0.5:
        applicable.append("salary expectation aligns with senior IC band")
    jd_ct = _safe_int(f, 'jd_skill_count')
    if jd_ct >= 4:
        applicable.append("broad skill overlap across " + str(jd_ct) + " JD requirements")
    if not applicable:
        return ""
    return applicable[h % len(applicable)]
