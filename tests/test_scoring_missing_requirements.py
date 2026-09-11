import numpy as np
import pytest
from human_ai_codesign.pipeline import CoDesignScorer

# Helper scorer instance
scorer = CoDesignScorer()

def test_score_skills_empty_requirements():
    # Empty required skills should return np.nan
    result = scorer.score_skills([], ["python", "ml"], "any")
    assert np.isnan(result)

def test_score_education_empty_requirements():
    # Empty education requirements should return np.nan
    result = scorer.score_education([], ["PhD", "MSc"], "any")
    assert np.isnan(result)

def test_score_experience_no_requirements_no_keywords():
    # No years requirement and no project keywords should yield np.nan
    result = scorer.score_experience(None, None, {"data_analytics": 3}, "Some job description")
    assert np.isnan(result)

def test_score_experience_no_years_with_keyword():
    # No years requirement but project keyword present should score 1.0
    result = scorer.score_experience(None, None, {"data_analytics": 0}, "We need healthcare experience")
    assert result == 1.0

def test_score_experience_years_only():
    # Years requirement provided, candidate meets it fully
    result = scorer.score_experience(2, None, {"data_analytics": 3}, "No keyword here")
    assert result == 1.0

def test_score_experience_years_and_keyword_combined():
    # Both years and keyword present, combine with weights 0.7 and 0.3
    result = scorer.score_experience(4, None, {"data_analytics": 2}, "Experience with big data projects")
    # y_score = 2/4 = 0.5, proj_score = 1.0 -> combined = 0.7*0.5 + 0.3*1.0 = 0.35 + 0.3 = 0.65
    assert pytest.approx(result, 0.0001) == 0.65

def test_score_salary_job_type_none():
    # Neither salary nor job_type provided
    extraction = {"salary": None, "job_type": None}
    result = scorer.score_salary_job_type(extraction)
    assert np.isnan(result)

def test_score_salary_job_type_full_time():
    extraction = {"salary": None, "job_type": "Full-time"}
    result = scorer.score_salary_job_type(extraction)
    assert result == 1.0

def test_score_salary_job_type_non_full():
    extraction = {"salary": None, "job_type": "Part-time"}
    result = scorer.score_salary_job_type(extraction)
    assert np.isnan(result)
