"""
Honeypot detection - identifies synthetic/impossible candidate profiles.
"""


def is_honeypot(f):
    if f.timeline_impossible:
        return True
    if f.expert_zero_usage_count > 3:
        return True
    if f.expert_zero_usage_count >= 2 and f.skill_career_coherence < 0.15:
        return True
    return False
