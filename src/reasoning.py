from src.features import CandidateFeatures


def generate_reasoning(f: CandidateFeatures, rank: int) -> str:
    parts = []

    # Core identity — always present, uses actual profile values
    parts.append(f"{f.current_title} with {f.yoe:.1f}yr experience")

    # Best career signal
    if f.ai_ml_months >= 12:
        parts.append(f"{f.ai_ml_months // 12}yr applied AI/ML in production roles")
    elif f.has_product_company:
        parts.append("product company background")

    # Top matched skill against JD requirements
    if f.top_matched_skill:
        parts.append(f"strong match on {f.top_matched_skill}")

    # Positive behavioral signals
    if f.open_to_work:
        parts.append("actively open to work")
    if f.github_activity_score > 60:
        parts.append(f"active GitHub (score {f.github_activity_score:.0f})")

    # Honest concerns — only surfaced at lower ranks for rank-consistency
    if rank > 20:
        concerns = []
        if f.notice_period_days > 90:
            concerns.append(f"{f.notice_period_days}d notice period")
        if f.recruiter_response_rate < 0.3:
            concerns.append("low recruiter response rate")
        if f.is_consulting_only:
            concerns.append("consulting-only background")
        if not f.open_to_work:
            concerns.append("not marked open-to-work")
        if concerns:
            parts.append(f"concern: {'; '.join(concerns)}")

    return "; ".join(parts) + "."
