import numpy as np

from human_ai_codesign.pipeline import MATCHING_WEIGHTS, apply_human_decision
from human_ai_codesign.ui_helpers import (
    ACCEPT_RECOMMENDATION,
    APPLICATION_LINK_UNAVAILABLE,
    build_match_evidence_rows,
    coverage_aware_tier_label,
    decision_key,
    format_component_score,
    get_human_decision,
    job_url_or_none,
    match_score_label,
    persist_human_decisions,
)


def test_missing_component_score_displays_not_evaluated():
    assert format_component_score(np.nan) == "Not evaluated"
    assert format_component_score(None) == "Not evaluated"
    assert format_component_score(0.75) == "75.0%"


def test_match_score_label_reflects_partial_coverage():
    assert match_score_label(75.0) == "Match on Evaluated Factors"
    assert match_score_label(99.9) == "Match on Evaluated Factors"
    assert match_score_label(100.0) == "Match Score"


def test_coverage_aware_tier_label_marks_limited_evidence_without_changing_tier_value():
    assert coverage_aware_tier_label("Strong match", 20.0) == "Strong match on evaluated factors - Limited evidence"
    assert coverage_aware_tier_label("Strong match", 100.0) == "Strong match"


def test_evidence_rows_use_existing_scores_without_recomputing():
    row = {
        "title": "Healthcare Data Scientist",
        "source": "Indeed",
        "date_retrieved": "2026-09-11",
        "required_skills": ["sql"],
        "matched_skills": ["sql"],
        "missing_skills": [],
        "skills_score": 0.42,
        "experience_projects_score": np.nan,
        "education_score": np.nan,
        "role_alignment_score": 0.91,
        "location_work_score": 1.0,
        "salary_job_type_score": np.nan,
        "location": "Remote",
        "work_mode": "Remote",
        "min_experience": None,
    }
    profile = {
        "education": ["MS Data Science and Analytics"],
        "preferred_roles": ["Healthcare Data Scientist"],
        "preferred_locations": ["Remote"],
        "preferred_work_arrangements": ["Remote", "Hybrid"],
    }

    evidence = build_match_evidence_rows(row, profile)
    by_factor = {item["Factor"]: item for item in evidence}

    assert by_factor["Skills"]["Score"] == "42.0%"
    assert "matched: sql" in by_factor["Skills"]["Evidence Used"]
    assert by_factor["Skills"]["Source"] == "Job posting (Indeed, retrieved 2026-09-11); Candidate Profile"
    assert by_factor["Experience / Projects"]["Evidence Used"] == "Not enough information"
    assert by_factor["Experience / Projects"]["Score"] == "Not evaluated"
    assert by_factor["Salary / Job Type"]["Score"] == "Not evaluated"


def test_accept_recommendation_matches_legacy_accept_ai():
    assert apply_human_decision("Apply", ACCEPT_RECOMMENDATION) == "Apply"
    assert apply_human_decision("Consider", "Accept AI") == "Consider"


def test_human_review_decision_persists_by_job_key_across_filtered_rows():
    key = decision_key("Healthcare Data Analyst", "CareMetrics")
    decisions = {}
    persist_human_decisions(decisions, [{"_decision_key": key, "Human Review Decision": "Needs Manual Review"}])

    assert get_human_decision(decisions, key) == "Needs Manual Review"
    assert get_human_decision(decisions, decision_key("Other", "Company")) == ACCEPT_RECOMMENDATION


def test_legacy_accept_ai_is_normalized_when_persisted():
    key = decision_key("Data Analyst", "Metro Analytics")
    decisions = {}
    persist_human_decisions(decisions, [{"_decision_key": key, "Human Review Decision": "Accept AI"}])
    assert decisions[key] == ACCEPT_RECOMMENDATION


def test_job_url_display_decision():
    assert job_url_or_none({"job_url": "https://example.com/job/123"}) == "https://example.com/job/123"
    assert job_url_or_none({"job_url": ""}) is None
    assert job_url_or_none({}) is None
    assert APPLICATION_LINK_UNAVAILABLE == "Application link unavailable in this evaluation dataset."


def test_ui_helpers_do_not_change_scoring_weights():
    assert list(MATCHING_WEIGHTS.values()) == [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
