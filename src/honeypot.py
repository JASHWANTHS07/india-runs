from src.features import CandidateFeatures


def is_honeypot(f: CandidateFeatures) -> bool:
    if f.timeline_impossible:
        return True
    if f.expert_zero_usage_count > 3:
        return True
    return False
