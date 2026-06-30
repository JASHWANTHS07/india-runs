"""
Honeypot detection - identifies synthetic/impossible candidate profiles.
"""


def is_honeypot(f):
    # Original conditions
    if f.timeline_impossible:
        return True
    if f.expert_zero_usage_count > 3:
        return True
    if f.expert_zero_usage_count >= 2 and f.skill_career_coherence < 0.15:
        return True

    # NEW: inverted salary range (min > max)
    if getattr(f, 'salary_inverted', False):
        return True

    # NEW: assessment contradicts claimed proficiency
    if (getattr(f, 'assessment_proficiency_gap', 0) > 2.0
            and getattr(f, 'assessment_count', 0) >= 2):
        return True

    # NEW: template summary + career description mismatch
    if (getattr(f, 'summary_is_template', False)
            and getattr(f, 'career_desc_title_mismatch_count', 0) >= 2):
        return True

    # NEW: endorsement anomaly
    if (f.endorsements_total > 5 * f.connection_count
            and f.connection_count < 10):
        return True

    # NEW: ghost profile — complete but zero activity
    if (f.profile_completeness >= 80 and f.days_since_active > 180
            and f.applications_30d == 0 and f.profile_views_30d == 0):
        return True

    return False
