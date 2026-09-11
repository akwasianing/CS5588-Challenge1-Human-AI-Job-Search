from urllib.parse import urlparse

import pandas as pd


ACCEPT_RECOMMENDATION = "Accept Recommendation"
APPLICATION_LINK_UNAVAILABLE = "Application link unavailable in this evaluation dataset."


def format_component_score(value):
    if pd.isna(value):
        return "Not evaluated"
    return f"{float(value) * 100:.1f}%"


def match_score_label(coverage_percent):
    if pd.notna(coverage_percent) and float(coverage_percent) < 100.0:
        return "Match on Evaluated Factors"
    return "Match Score"


def coverage_aware_tier_label(match_tier, coverage_percent):
    if pd.notna(coverage_percent) and float(coverage_percent) < 100.0:
        return f"{match_tier} on evaluated factors - Limited evidence"
    return match_tier


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _join_or_none(values):
    clean = [str(v).strip() for v in _as_list(values) if str(v).strip()]
    return ", ".join(clean) if clean else "None"


def _has_value(value):
    if isinstance(value, list):
        return bool(value)
    if value is None:
        return False
    return not pd.isna(value)


def _job_source(row):
    source = row.get("source")
    date_retrieved = row.get("date_retrieved")
    if _has_value(source) and _has_value(date_retrieved):
        return f"Job posting ({source}, retrieved {date_retrieved})"
    if _has_value(source):
        return f"Job posting ({source})"
    return "Job record / description"


def build_match_evidence_rows(row, candidate_profile):
    job_source = _job_source(row)
    candidate_source = "Candidate Profile"

    required_skills = _as_list(row.get("required_skills", []))
    matched_skills = _as_list(row.get("matched_skills", []))
    missing_skills = _as_list(row.get("missing_skills", []))
    skills_evidence = "Not enough information"
    if not pd.isna(row.get("skills_score")):
        skills_evidence = (
            f"Required: {_join_or_none(required_skills)}; "
            f"matched: {_join_or_none(matched_skills)}; "
            f"missing: {_join_or_none(missing_skills)}"
        )

    min_experience = row.get("min_experience")
    experience_evidence = "Not enough information"
    if not pd.isna(row.get("experience_projects_score")):
        requirement = f"{min_experience:g}+ years" if _has_value(min_experience) else "Requirement identified in job description"
        experience_evidence = f"{requirement}; candidate experience profile available"

    education_evidence = "Not enough information"
    if not pd.isna(row.get("education_score")):
        education_evidence = f"Candidate education: {_join_or_none(candidate_profile.get('education', []))}"

    role_evidence = (
        f"Job title: {row.get('title', 'Not specified')}; "
        f"preferred roles: {_join_or_none(candidate_profile.get('preferred_roles', []))}"
    )

    location = row.get("location") if _has_value(row.get("location")) else "Not specified"
    work_mode = row.get("work_mode") if _has_value(row.get("work_mode")) else "Not specified"
    location_evidence = (
        f"Job: {location} / {work_mode}; "
        f"candidate prefers locations {_join_or_none(candidate_profile.get('preferred_locations', []))} "
        f"and work modes {_join_or_none(candidate_profile.get('preferred_work_arrangements', []))}"
    )

    salary_evidence = "Not enough information"
    if not pd.isna(row.get("salary_job_type_score")):
        salary_evidence = "Salary/job type information available in job record and candidate preferences"

    return [
        {
            "Factor": "Skills",
            "Evidence Used": skills_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("skills_score")),
        },
        {
            "Factor": "Experience / Projects",
            "Evidence Used": experience_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("experience_projects_score")),
        },
        {
            "Factor": "Education",
            "Evidence Used": education_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("education_score")),
        },
        {
            "Factor": "Role Alignment",
            "Evidence Used": role_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("role_alignment_score")),
        },
        {
            "Factor": "Location / Work Arrangement",
            "Evidence Used": location_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("location_work_score")),
        },
        {
            "Factor": "Salary / Job Type",
            "Evidence Used": salary_evidence,
            "Source": f"{job_source}; {candidate_source}",
            "Score": format_component_score(row.get("salary_job_type_score")),
        },
    ]


def decision_key(title, company):
    return f"{title} @ {company}"


def normalize_human_decision(value):
    if value == "Accept AI":
        return ACCEPT_RECOMMENDATION
    return value


def get_human_decision(decisions, key):
    return normalize_human_decision(decisions.get(key, ACCEPT_RECOMMENDATION))


def persist_human_decisions(decisions, edited_rows):
    for row in edited_rows:
        key = row.get("_decision_key")
        if key:
            decisions[key] = normalize_human_decision(row.get("Human Review Decision"))
    return decisions


def valid_job_url(value):
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def job_url_or_none(row):
    value = row.get("job_url")
    return value.strip() if valid_job_url(value) else None
