import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "l&t infotech", "niit technologies", "ibm india",
}

AI_ML_KEYWORDS = {
    "embedding", "vector", "retrieval", "ranking", "recommendation", "nlp",
    "llm", "rag", "fine-tun", "transformer", "bert", "gpt", "neural",
    "deep learning", "machine learning", " ml ", "search", "semantic",
    "similarity", "classification", "pytorch", "tensorflow", "keras",
    "huggingface", "sentence-transformer", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "elasticsearch", "opensearch", "feature engineering",
    "model training", "model deployment", "mlops", "xgboost", "lightgbm",
}

PRODUCTION_KEYWORDS = {
    "deployed", "production", "shipped", "built at scale", "serving",
    "launched", "released in", "implemented in production", "live system",
}

VECTOR_SEARCH_KEYWORDS = {
    "vector", "embedding", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "semantic search", "dense retrieval",
    "approximate nearest neighbor", "ann", "hnsw",
}

JD_CORE_SKILLS = [
    "embeddings", "sentence-transformers", "vector database", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "hybrid search",
    "retrieval", "ranking", "ndcg", "mrr", "map", "python", "llm", "fine-tuning",
    "lora", "qlora", "peft", "learning-to-rank", "xgboost", "lightgbm",
    "information retrieval", "semantic search", "recommendation", "search",
    "pytorch", "tensorflow", "huggingface", "transformer",
]

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

REFERENCE_DATE = datetime(2026, 6, 27)


@dataclass
class CandidateFeatures:
    candidate_id: str
    # Profile
    yoe: float
    current_title: str
    current_company: str
    # Career
    has_product_company: bool
    is_consulting_only: bool
    ai_ml_months: int
    shipped_count: int
    vector_search_experience: bool
    # Skills
    top_matched_skill: Optional[str]
    skills_match_score: float
    # Education
    best_education_tier: int
    # Location
    country: str
    city: str
    in_preferred_india_city: bool
    willing_to_relocate: bool
    # Behavioral
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
    # Honeypot flags
    timeline_impossible: bool
    expert_zero_usage_count: int
    # Embedding input
    profile_text: str = ""


def _company_is_consulting(name: str) -> bool:
    name_lower = name.lower()
    return any(firm in name_lower for firm in CONSULTING_FIRMS)


def _text_has_any(text: str, keywords: set) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def extract_features(candidate: dict) -> CandidateFeatures:
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    education = candidate.get("education") or []
    signals = candidate.get("redrob_signals") or {}

    # --- Career analysis ---
    companies = [r.get("company") or "" for r in career]
    is_consulting_only = bool(companies) and all(_company_is_consulting(c) for c in companies)
    has_product_company = any(not _company_is_consulting(c) for c in companies)

    ai_ml_months = 0
    shipped_count = 0
    vector_search_experience = False
    for role in career:
        desc = (role.get("description") or "").lower()
        duration = int(role.get("duration_months") or 0)
        has_ai = _text_has_any(desc, AI_ML_KEYWORDS)
        has_prod = _text_has_any(desc, PRODUCTION_KEYWORDS)
        has_vec = _text_has_any(desc, VECTOR_SEARCH_KEYWORDS)
        if has_ai:
            ai_ml_months += duration
        if has_ai and has_prod:
            shipped_count += 1
        if has_vec:
            vector_search_experience = True

    # --- Skills matching ---
    total_score = 0.0
    top_matched_skill: Optional[str] = None
    top_weight = 0.0
    for skill in skills:
        name = (skill.get("name") or "").lower()
        prof = PROFICIENCY_WEIGHTS.get(skill.get("proficiency") or "beginner", 0.25)
        for jd_skill in JD_CORE_SKILLS:
            if jd_skill.lower() in name or name in jd_skill.lower():
                total_score += prof
                if prof > top_weight:
                    top_weight = prof
                    top_matched_skill = skill.get("name")
                break
    # Normalize: assume hitting 30% of JD skills = score of 1.0
    skills_match_score = min(1.0, total_score / max(1.0, len(JD_CORE_SKILLS) * 0.3))

    # --- Education ---
    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 4}
    tiers = [tier_map.get(e.get("tier") or "unknown", 4) for e in education]
    best_education_tier = min(tiers) if tiers else 4

    # --- Location ---
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    in_preferred = any(city in location for city in INDIA_PREFERRED_CITIES)
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    # --- Behavioral ---
    open_to_work = bool(signals.get("open_to_work_flag", False))
    last_active = signals.get("last_active_date") or "2020-01-01"
    try:
        last_dt = datetime.strptime(last_active, "%Y-%m-%d")
        days_since_active = max(0, (REFERENCE_DATE - last_dt).days)
    except (ValueError, TypeError):
        days_since_active = 365

    recruiter_response_rate = float(signals.get("recruiter_response_rate") or 0.5)
    github_score = float(signals.get("github_activity_score") if signals.get("github_activity_score") is not None else -1)
    notice_period = int(signals.get("notice_period_days") or 60)
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    salary_min = float(salary.get("min") or 0)
    salary_max = float(salary.get("max") or 0)
    profile_completeness = float(signals.get("profile_completeness_score") or 50)
    applications_30d = int(signals.get("applications_submitted_30d") or 0)
    interview_rate = float(signals.get("interview_completion_rate") or 0.5)

    # --- Honeypot detection ---
    stated_yoe = float(profile.get("years_of_experience") or 0)
    total_career_months = sum(int(r.get("duration_months") or 0) for r in career)
    timeline_impossible = stated_yoe > (total_career_months / 12.0 + 5.0)

    expert_zero_count = sum(
        1 for s in skills
        if (s.get("proficiency") or "") == "expert"
        and int(s.get("duration_months") or 0) == 0
        and int(s.get("endorsements") or 0) == 0
    )

    # --- Profile text for embedding ---
    text_parts = [
        profile.get("headline") or "",
        profile.get("summary") or "",
        " ".join(s.get("name") or "" for s in skills),
    ]
    for role in career:
        text_parts.append(role.get("title") or "")
        text_parts.append(role.get("description") or "")
    profile_text = " ".join(p for p in text_parts if p).strip()[:2000]  # cap for embedding

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        yoe=stated_yoe,
        current_title=profile.get("current_title") or "",
        current_company=profile.get("current_company") or "",
        has_product_company=has_product_company,
        is_consulting_only=is_consulting_only,
        ai_ml_months=ai_ml_months,
        shipped_count=shipped_count,
        vector_search_experience=vector_search_experience,
        top_matched_skill=top_matched_skill,
        skills_match_score=skills_match_score,
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
        timeline_impossible=timeline_impossible,
        expert_zero_usage_count=expert_zero_count,
        profile_text=profile_text,
    )
