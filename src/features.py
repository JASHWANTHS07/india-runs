import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "l&t infotech", "niit technologies", "ibm india",
}

_TITLE_KEYWORDS_T4 = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "search engineer",
    "recommendation systems engineer", "ranking engineer",
    "applied ml engineer", "applied scientist", "research scientist",
    "deep learning engineer", "ml scientist", "ai specialist",
    "ai research engineer", "lead ai engineer", "staff machine learning",
    "senior machine learning", "senior ai engineer", "senior ml engineer",
    "senior nlp engineer", "senior data scientist", "senior applied scientist",
    "senior software engineer (ml)",
    "machine learning", "artificial intelligence",
]

_TITLE_KEYWORDS_T3 = [
    "software engineer", "backend engineer", "data engineer",
    "full stack developer", "python developer", "platform engineer",
    "senior software engineer", "senior data engineer",
    "senior backend engineer", "systems engineer",
]

_TITLE_KEYWORDS_T2 = [
    "junior ml engineer", "junior data scientist", "junior ai engineer",
    "devops engineer", "cloud engineer", "frontend engineer",
    "qa engineer", "mobile developer", "java developer",
    ".net developer", "ios developer", "android developer",
    "embedded engineer", "infrastructure engineer",
    "security engineer", "site reliability engineer", "sre",
]

_TITLE_CV_ONLY = [
    "computer vision engineer", "cv engineer", "image processing engineer",
    "robotics engineer", "speech engineer",
]

_TITLE_NON_TECH = {
    "hr manager", "marketing manager", "sales executive", "accountant",
    "operations manager", "customer support", "graphic designer",
    "content writer", "project manager", "business analyst",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer",
}


def _compute_title_tier(title):
    t = title.lower().strip()
    for kw in _TITLE_KEYWORDS_T4:
        if kw in t:
            return 4
    for kw in _TITLE_KEYWORDS_T3:
        if kw in t:
            return 3
    for kw in _TITLE_KEYWORDS_T2:
        if kw in t:
            return 2
    for kw in _TITLE_NON_TECH:
        if kw in t:
            return 0
    if "engineer" in t or "developer" in t or "architect" in t:
        return 2
    return 0


def _is_cv_only_title(title):
    t = title.lower()
    return any(kw in t for kw in _TITLE_CV_ONLY)


AI_ML_KEYWORDS = {
    "embedding", "vector", "retrieval", "ranking", "recommendation", "nlp",
    "llm", "rag", "fine-tun", "transformer", "bert", "gpt", "neural",
    "deep learning", "machine learning", " ml ", "search", "semantic",
    "similarity", "classification", "pytorch", "tensorflow", "keras",
    "huggingface", "sentence-transformer", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "elasticsearch", "opensearch", "feature engineering",
    "model training", "model deployment", "mlops", "xgboost", "lightgbm",
    "natural language", "text mining", "tokeniz", "named entity",
    "sentiment", "information retrieval", "dense retrieval", "re-rank",
    "learning-to-rank", "learning to rank", "a/b test", "ndcg", "mrr",
    "recommendation system", "collaborative filtering", "content-based",
}

RETRIEVAL_RANKING_KEYWORDS = {
    "ranking", "retrieval", "search", "recommendation",
    "information retrieval", "learning-to-rank", "learning to rank",
    "ndcg", "mrr", "bm25", "inverted index", "query",
    "relevance", "reranking", "re-ranking", "candidate generation",
    "two-tower", "bi-encoder", "cross-encoder", "dense retrieval",
    "hybrid search", "search quality", "search relevance",
    "embedding-based retrieval", "semantic search",
    "recommendation system", "collaborative filtering",
    "discovery feed", "personalization",
}

PRODUCTION_KEYWORDS = {
    "deployed", "production", "shipped", "built at scale", "serving",
    "launched", "released in", "implemented in production", "live system",
    "real users", "a/b test", "online experiment",
}

VECTOR_SEARCH_KEYWORDS = {
    "vector", "embedding", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "semantic search", "dense retrieval",
    "approximate nearest neighbor", "ann", "hnsw",
}

JD_SKILLS_TIER1 = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "vector search", "vector db",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "elastic search",
    "hybrid search", "retrieval", "ranking", "re-ranking",
    "information retrieval", "semantic search", "dense retrieval",
    "recommendation", "recommendation systems", "search",
    "search infrastructure", "learning-to-rank", "learning to rank",
    "bm25", "ndcg", "mrr", "python",
}

JD_SKILLS_TIER2 = {
    "pytorch", "tensorflow", "keras",
    "huggingface", "hugging face", "hugging face transformers",
    "transformer", "bert", "gpt",
    "xgboost", "lightgbm", "gradient boosting",
    "llm", "llms", "large language model",
    "fine-tuning", "fine tuning", "finetuning",
    "fine-tuning llms",
    "lora", "qlora", "peft",
    "machine learning", "deep learning",
    "mlops", "mlflow", "kubeflow", "feature engineering",
    "model training", "model deployment",
    "nlp", "natural language processing",
    "scikit-learn", "sklearn",
    "rag", "langchain",
    "data science", "prompt engineering",
}

JD_CORE_SKILLS = list(JD_SKILLS_TIER1 | JD_SKILLS_TIER2)

INDIA_PREFERRED_CITIES = {
    "pune", "noida", "delhi", "gurgaon", "gurugram", "bengaluru", "bangalore",
    "hyderabad", "mumbai", "chennai", "new delhi", "delhi ncr",
}

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25,
}

DEGREE_LEVEL_KEYWORDS = {
    4: ["phd", "ph.d", "doctorate", "doctoral"],
    3: ["master", "m.tech", "m.s.", "m.sc", "mtech", "ms ", "meng", "m.eng"],
    2: ["bachelor", "b.tech", "b.e.", "b.sc", "btech", "bs ", "b.s.", "ba "],
    1: ["diploma", "associate", "certificate"],
}

CS_FIELD_KEYWORDS = {
    "computer science", "computer engineering", "software engineering",
    "information technology", "mathematics", "statistics", "physics",
    "electrical engineering", "electronics",
}

AI_FIELD_KEYWORDS = {
    "artificial intelligence", "machine learning", "data science",
    "deep learning", "natural language processing", "computational linguistics",
}

ML_CERT_KEYWORDS = {
    "aws", "amazon", "gcp", "google cloud", "azure", "microsoft certified",
    "tensorflow", "pytorch", "keras", "deep learning", "machine learning",
    "kubernetes", "docker", "mlops", "data engineer", "data science",
    "nvidia", "databricks", "snowflake", "apache spark",
}

COMPANY_SIZE_ORD = {
    "1-10": 0, "11-50": 1, "51-200": 2, "201-500": 3,
    "501-1000": 4, "1001-5000": 5, "5001-10000": 6, "10001+": 7,
}

WORK_MODE_SCORE = {
    "hybrid": 1.0, "flexible": 1.0,
    "onsite": 0.7, "remote": 0.3,
}

REFERENCE_DATE = datetime(2026, 6, 27)


@dataclass
class CandidateFeatures:
    candidate_id: str
    yoe: float
    current_title: str
    current_company: str
    title_relevance_tier: int
    is_cv_only_title: bool
    has_product_company: bool
    is_consulting_only: bool
    ai_ml_months: int
    shipped_count: int
    vector_search_experience: bool
    career_retrieval_months: int
    career_ai_depth_ratio: float
    has_product_ai_career: bool
    top_matched_skill: str
    skills_match_score: float
    jd_skill_count: int
    jd_tier1_skill_count: int
    skill_career_coherence: float
    non_tech_title_with_ai_skills: bool
    jd_skill_assessment_avg: float
    best_education_tier: int
    country: str
    city: str
    in_preferred_india_city: bool
    willing_to_relocate: bool
    open_to_work: bool
    last_active_date: str
    days_since_active: int
    recruiter_response_rate: float
    github_activity_score: float
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    profile_completeness: float
    applications_30d: int
    interview_completion_rate: float
    saved_by_recruiters: int
    profile_views_30d: int
    avg_response_time_hours: float
    verified_count: int
    endorsements_total: int
    connection_count: int
    offer_acceptance_rate: float
    consulting_months: int
    product_months: int
    technical_yoe: float
    timeline_impossible: bool
    expert_zero_usage_count: int
    has_cs_degree: bool
    highest_degree_level: int
    education_ai_relevance: float
    education_recency: int
    cert_count: int
    ml_cert_count: int
    cert_recency: int
    ai_title_count: int
    title_progression: int
    avg_tenure_months: float
    num_roles: int
    max_company_size_ord: int
    current_company_size_ord: int
    total_skill_count: int
    avg_skill_proficiency: float
    endorsed_skill_ratio: float
    skill_keyword_density: float
    work_mode_match: float
    search_appearance_30d: int
    salary_range_width: float
    platform_tenure_days: int
    profile_text: str = ""


def _company_is_consulting(name):
    name_lower = name.lower()
    return any(firm in name_lower for firm in CONSULTING_FIRMS)


def _text_has_any(text, keywords):
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _count_keyword_hits(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _skill_matches_jd(skill_name_lower):
    for jd_skill in JD_SKILLS_TIER1:
        if jd_skill in skill_name_lower or skill_name_lower in jd_skill:
            return True, 1
    for jd_skill in JD_SKILLS_TIER2:
        if jd_skill in skill_name_lower or skill_name_lower in jd_skill:
            return True, 2
    return False, 0


def extract_features(candidate):
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    education = candidate.get("education") or []
    signals = candidate.get("redrob_signals") or {}
    current_title = profile.get("current_title") or ""

    title_relevance_tier = _compute_title_tier(current_title)
    is_cv_only = _is_cv_only_title(current_title)

    companies = [r.get("company") or "" for r in career]
    is_consulting_only = bool(companies) and all(_company_is_consulting(c) for c in companies)
    has_product_company = any(not _company_is_consulting(c) for c in companies)

    ai_ml_months = 0
    shipped_count = 0
    vector_search_experience = False
    career_retrieval_months = 0
    product_ai_months = 0
    total_career_months = 0
    consulting_months = 0
    product_months = 0

    for role in career:
        desc = (role.get("description") or "").lower()
        duration = int(role.get("duration_months") or 0)
        total_career_months += duration
        company = role.get("company") or ""
        is_product_co = not _company_is_consulting(company)
        if is_product_co:
            product_months += duration
        else:
            consulting_months += duration
        has_ai = _text_has_any(desc, AI_ML_KEYWORDS)
        has_prod = _text_has_any(desc, PRODUCTION_KEYWORDS)
        has_vec = _text_has_any(desc, VECTOR_SEARCH_KEYWORDS)
        has_retrieval = _text_has_any(desc, RETRIEVAL_RANKING_KEYWORDS)
        if has_ai:
            ai_ml_months += duration
            if is_product_co:
                product_ai_months += duration
        if has_ai and has_prod:
            shipped_count += 1
        if has_vec:
            vector_search_experience = True
        if has_retrieval:
            career_retrieval_months += duration

    career_ai_depth_ratio = ai_ml_months / max(1, total_career_months)
    has_product_ai_career = product_ai_months > 12
    technical_yoe = product_months / 12.0

    ai_title_count = 0
    max_company_size_ord = 0
    role_tiers = []
    durations = []
    for role in career:
        role_title = role.get("title") or ""
        if _compute_title_tier(role_title) == 4:
            ai_title_count += 1
        role_size = COMPANY_SIZE_ORD.get(role.get("company_size") or "", 0)
        if role_size > max_company_size_ord:
            max_company_size_ord = role_size
        role_tiers.append((_compute_title_tier(role_title), role.get("start_date") or ""))
        durations.append(int(role.get("duration_months") or 0))

    num_roles = len(career)
    avg_tenure_months = sum(durations) / max(1, num_roles)

    if len(role_tiers) >= 2:
        sorted_tiers = sorted(role_tiers, key=lambda x: x[1])
        first_tier = sorted_tiers[0][0]
        last_tier = sorted_tiers[-1][0]
        diff = last_tier - first_tier
        title_progression = 1 if diff > 0 else (-1 if diff < 0 else 0)
    else:
        title_progression = 0

    current_company_size_ord = COMPANY_SIZE_ORD.get(
        profile.get("current_company_size") or "", 0
    )

    total_score = 0.0
    top_matched_skill = None
    top_weight = 0.0
    jd_skill_count = 0
    jd_tier1_count = 0
    matched_skill_names = []
    skill_assessments = signals.get("skill_assessment_scores") or {}

    for skill in skills:
        name = skill.get("name") or ""
        name_lower = name.lower()
        matched, tier = _skill_matches_jd(name_lower)
        if not matched:
            continue
        jd_skill_count += 1
        if tier == 1:
            jd_tier1_count += 1
        matched_skill_names.append(name_lower)
        prof = PROFICIENCY_WEIGHTS.get(skill.get("proficiency") or "beginner", 0.25)
        duration_mo = int(skill.get("duration_months") or 0)
        dur_mult = min(1.0, 0.3 + 0.7 * duration_mo / 24) if duration_mo > 0 else 0.4
        endorse = int(skill.get("endorsements") or 0)
        end_mult = min(1.0, 0.5 + 0.5 * endorse / 10)
        tier_mult = 1.0 if tier == 1 else 0.6
        weight = prof * dur_mult * end_mult * tier_mult
        total_score += weight
        if weight > top_weight:
            top_weight = weight
            top_matched_skill = name

    max_possible = len(JD_CORE_SKILLS) * 0.25
    skills_match_score = min(1.0, total_score / max(1.0, max_possible))

    total_skill_count = len(skills)
    if total_skill_count > 0:
        prof_sum = sum(
            PROFICIENCY_WEIGHTS.get(s.get("proficiency") or "beginner", 0.25)
            for s in skills
        )
        avg_skill_proficiency = prof_sum / total_skill_count
        endorsed_count = sum(1 for s in skills if int(s.get("endorsements") or 0) > 0)
        endorsed_skill_ratio = endorsed_count / total_skill_count
    else:
        avg_skill_proficiency = 0.25
        endorsed_skill_ratio = 0.0
    skill_keyword_density = jd_skill_count / max(1, total_skill_count)

    jd_assessment_scores = []
    for skill_name, score in skill_assessments.items():
        sn = skill_name.lower()
        matched_a, _ = _skill_matches_jd(sn)
        if matched_a:
            jd_assessment_scores.append(score)
    jd_skill_assessment_avg = (
        sum(jd_assessment_scores) / len(jd_assessment_scores)
        if jd_assessment_scores else -1.0
    )

    all_desc_text = " ".join((r.get("description") or "") for r in career).lower()
    career_ai_keyword_count = _count_keyword_hits(all_desc_text, AI_ML_KEYWORDS)
    if jd_skill_count <= 1:
        skill_career_coherence = 0.5
    elif career_ai_keyword_count == 0:
        if jd_skill_count >= 3:
            skill_career_coherence = 0.05
        else:
            skill_career_coherence = 0.25
    else:
        ratio = career_ai_keyword_count / (jd_skill_count + 1)
        skill_career_coherence = min(1.0, ratio * 0.4 + 0.3)

    non_tech_with_ai = (title_relevance_tier == 0 and jd_skill_count >= 3)

    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 4}
    tiers = [tier_map.get(e.get("tier") or "unknown", 4) for e in education]
    best_education_tier = min(tiers) if tiers else 4

    has_cs_degree = False
    highest_degree_level = 0
    education_ai_relevance = 0.0
    latest_end_year = 0
    for edu in education:
        field = (edu.get("field_of_study") or "").lower()
        degree = (edu.get("degree") or "").lower()
        end_year = int(edu.get("end_year") or 0)
        if any(kw in field for kw in CS_FIELD_KEYWORDS):
            has_cs_degree = True
        if any(kw in field for kw in AI_FIELD_KEYWORDS):
            education_ai_relevance = max(education_ai_relevance, 1.0)
        elif any(kw in field for kw in CS_FIELD_KEYWORDS):
            education_ai_relevance = max(education_ai_relevance, 0.5)
        for level, keywords in DEGREE_LEVEL_KEYWORDS.items():
            if any(kw in degree for kw in keywords):
                highest_degree_level = max(highest_degree_level, level)
                break
        if end_year > latest_end_year:
            latest_end_year = end_year
    education_recency = (2026 - latest_end_year) if latest_end_year > 0 else 20

    certs = candidate.get("certifications") or []
    cert_count = len(certs)
    ml_cert_count = 0
    latest_cert_year = 0
    for cert in certs:
        cert_name = (cert.get("name") or "").lower()
        cert_issuer = (cert.get("issuer") or "").lower()
        cert_text = cert_name + " " + cert_issuer
        if any(kw in cert_text for kw in ML_CERT_KEYWORDS):
            ml_cert_count += 1
        cert_year = int(cert.get("year") or 0)
        if cert_year > latest_cert_year:
            latest_cert_year = cert_year
    cert_recency = (2026 - latest_cert_year) if latest_cert_year > 0 else 10

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    in_preferred = any(city in location for city in INDIA_PREFERRED_CITIES)
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    open_to_work = bool(signals.get("open_to_work_flag", False))
    last_active = signals.get("last_active_date") or "2020-01-01"
    try:
        last_dt = datetime.strptime(last_active, "%Y-%m-%d")
        days_since_active = max(0, (REFERENCE_DATE - last_dt).days)
    except (ValueError, TypeError):
        days_since_active = 365

    recruiter_response_rate = float(signals.get("recruiter_response_rate") or 0.5)
    github_score = float(
        signals.get("github_activity_score")
        if signals.get("github_activity_score") is not None
        else -1
    )
    notice_period = int(signals.get("notice_period_days") or 60)
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    salary_min = float(salary.get("min") or 0)
    salary_max = float(salary.get("max") or 0)
    profile_completeness = float(signals.get("profile_completeness_score") or 50)
    applications_30d = int(signals.get("applications_submitted_30d") or 0)
    interview_rate = float(signals.get("interview_completion_rate") or 0.5)

    saved_by_recruiters = int(signals.get("saved_by_recruiters_30d") or 0)
    profile_views_30d = int(signals.get("profile_views_received_30d") or 0)
    avg_response_time = float(signals.get("avg_response_time_hours") or 200)
    offer_accept = float(
        signals.get("offer_acceptance_rate")
        if signals.get("offer_acceptance_rate") is not None
        else -1
    )
    work_mode = (signals.get("preferred_work_mode") or "").lower()
    work_mode_match = WORK_MODE_SCORE.get(work_mode, 0.5)
    search_appearance_30d = int(signals.get("search_appearance_30d") or 0)
    salary_range_width = salary_max - salary_min
    signup_date_str = signals.get("signup_date") or ""
    try:
        signup_dt = datetime.strptime(signup_date_str, "%Y-%m-%d")
        platform_tenure_days = max(0, (REFERENCE_DATE - signup_dt).days)
    except (ValueError, TypeError):
        platform_tenure_days = 365
    endorsements_total_val = int(signals.get("endorsements_received") or 0)
    connection_count_val = int(signals.get("connection_count") or 0)
    verified_ct = (
        int(bool(signals.get("verified_email", False)))
        + int(bool(signals.get("verified_phone", False)))
        + int(bool(signals.get("linkedin_connected", False)))
    )

    stated_yoe = float(profile.get("years_of_experience") or 0)
    timeline_impossible = stated_yoe > (total_career_months / 12.0 + 5.0)
    expert_zero_count = sum(
        1 for s in skills
        if (s.get("proficiency") or "") == "expert"
        and int(s.get("duration_months") or 0) == 0
        and int(s.get("endorsements") or 0) == 0
    )

    text_parts = [
        profile.get("headline") or "",
        profile.get("summary") or "",
        " ".join(s.get("name") or "" for s in skills),
    ]
    for role in career:
        text_parts.append(role.get("title") or "")
        text_parts.append(role.get("description") or "")
    profile_text = " ".join(p for p in text_parts if p).strip()[:2000]

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        yoe=stated_yoe,
        current_title=current_title,
        current_company=profile.get("current_company") or "",
        title_relevance_tier=title_relevance_tier,
        is_cv_only_title=is_cv_only,
        has_product_company=has_product_company,
        is_consulting_only=is_consulting_only,
        ai_ml_months=ai_ml_months,
        shipped_count=shipped_count,
        vector_search_experience=vector_search_experience,
        career_retrieval_months=career_retrieval_months,
        career_ai_depth_ratio=career_ai_depth_ratio,
        has_product_ai_career=has_product_ai_career,
        top_matched_skill=top_matched_skill or "",
        skills_match_score=skills_match_score,
        jd_skill_count=jd_skill_count,
        jd_tier1_skill_count=jd_tier1_count,
        skill_career_coherence=skill_career_coherence,
        non_tech_title_with_ai_skills=non_tech_with_ai,
        jd_skill_assessment_avg=jd_skill_assessment_avg,
        best_education_tier=best_education_tier,
        country=country,
        city=location,
        in_preferred_india_city=in_preferred,
        willing_to_relocate=willing_to_relocate,
        open_to_work=open_to_work,
        last_active_date=last_active,
        days_since_active=days_since_active,
        recruiter_response_rate=recruiter_response_rate,
        github_activity_score=github_score,
        notice_period_days=notice_period,
        expected_salary_min=salary_min,
        expected_salary_max=salary_max,
        profile_completeness=profile_completeness,
        applications_30d=applications_30d,
        interview_completion_rate=interview_rate,
        saved_by_recruiters=saved_by_recruiters,
        profile_views_30d=profile_views_30d,
        avg_response_time_hours=avg_response_time,
        verified_count=verified_ct,
        endorsements_total=endorsements_total_val,
        connection_count=connection_count_val,
        offer_acceptance_rate=offer_accept,
        consulting_months=consulting_months,
        product_months=product_months,
        technical_yoe=technical_yoe,
        timeline_impossible=timeline_impossible,
        expert_zero_usage_count=expert_zero_count,
        has_cs_degree=has_cs_degree,
        highest_degree_level=highest_degree_level,
        education_ai_relevance=education_ai_relevance,
        education_recency=education_recency,
        cert_count=cert_count,
        ml_cert_count=ml_cert_count,
        cert_recency=cert_recency,
        ai_title_count=ai_title_count,
        title_progression=title_progression,
        avg_tenure_months=avg_tenure_months,
        num_roles=num_roles,
        max_company_size_ord=max_company_size_ord,
        current_company_size_ord=current_company_size_ord,
        total_skill_count=total_skill_count,
        avg_skill_proficiency=avg_skill_proficiency,
        endorsed_skill_ratio=endorsed_skill_ratio,
        skill_keyword_density=skill_keyword_density,
        work_mode_match=work_mode_match,
        search_appearance_30d=search_appearance_30d,
        salary_range_width=salary_range_width,
        platform_tenure_days=platform_tenure_days,
        profile_text=profile_text,
    )
