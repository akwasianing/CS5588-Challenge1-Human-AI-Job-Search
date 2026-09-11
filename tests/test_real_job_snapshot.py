import json

import pandas as pd
import pytest

import human_ai_codesign.pipeline as p


def test_evaluation_dataset_rows_remain_unchanged(monkeypatch):
    monkeypatch.setattr(p, "HAS_DATASETS", False)
    jobs = p.load_and_preprocess_dataset(data_source=p.EVALUATION_DATASET)

    assert len(jobs) == 8
    assert jobs["job_id"].tolist() == [f"J{i:03d}" for i in range(1, 9)]
    assert jobs.loc[jobs["job_id"] == "J006", "title"].iloc[0] == "Healthcare Data Analyst"
    assert jobs.loc[jobs["job_id"] == "J006", "company"].iloc[0] == "CareMetrics"


def test_evaluation_dataset_results_remain_unchanged(monkeypatch):
    monkeypatch.setattr(p, "HAS_DATASETS", False)
    results = p.run_full_codesign_pipeline(p.DEFAULT_CANDIDATE_PROFILE, top_n=20, data_source=p.EVALUATION_DATASET)

    assert results["title"].tolist() == [
        "Healthcare Data Analyst",
        "Business Intelligence Analyst",
        "Data Analyst",
        "Research Data Scientist",
        "Junior Data Scientist",
        "NLP Research Assistant",
        "Data Engineering Associate",
        "Machine Learning Engineer",
    ]
    assert results["match_percent"].tolist() == [100.0, 96.8, 96.7, 92.3, 88.1, 87.8, 82.4, 63.7]
    assert results["evaluation_coverage_percent"].tolist() == [75.0] * 8
    assert results["recommendation"].tolist() == [
        "Apply",
        "Apply",
        "Apply",
        "Apply",
        "Consider - Check Eligibility",
        "Apply",
        "Apply",
        "Review Eligibility First",
    ]


def test_real_job_json_loads_correctly(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {
        "snapshot_name": "Real Job Snapshot",
        "jobs": [
            {
                "job_id": "R001",
                "title": "Public Data Analyst",
                "company": "Example Health",
                "location": "Remote",
                "work_mode": "Remote",
                "min_experience": 2,
                "required_skills": ["sql", "python"],
                "preferred_skills": ["tableau"],
                "description": "Concise public-posting excerpt requiring SQL and Python.",
                "job_url": "https://example.com/jobs/r001",
                "source": "Example careers page",
                "date_retrieved": "2026-09-11"
            }
        ]
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)

    jobs = p.load_and_preprocess_dataset(data_source=p.REAL_JOB_SNAPSHOT)

    assert len(jobs) == 1
    row = jobs.iloc[0]
    assert row["job_id"] == "R001"
    assert row["job_url"] == "https://example.com/jobs/r001"
    assert row["source"] == "Example careers page"
    assert row["date_retrieved"] == "2026-09-11"
    assert row["required_skills_list"] == ["sql", "python"]
    assert row["preferred_skills_list"] == ["tableau"]


def test_real_job_result_passes_url_source_metadata(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {
        "jobs": [
            {
                "job_id": "R001",
                "title": "Public Data Analyst",
                "company": "Example Health",
                "location": "Remote",
                "work_mode": "Remote",
                "min_experience": 1,
                "required_skills": ["sql", "python"],
                "preferred_skills": ["tableau"],
                "description": "Concise public-posting excerpt requiring SQL and Python.",
                "job_url": "https://example.com/jobs/r001",
                "source": "Example careers page",
                "date_retrieved": "2026-09-11"
            }
        ]
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(p, "HAS_SBERT", False)

    results = p.run_full_codesign_pipeline(p.DEFAULT_CANDIDATE_PROFILE, top_n=1, data_source=p.REAL_JOB_SNAPSHOT)

    assert len(results) == 1
    result = results.iloc[0]
    assert result["job_url"] == "https://example.com/jobs/r001"
    assert result["source"] == "Example careers page"
    assert result["date_retrieved"] == "2026-09-11"


def test_populated_real_job_snapshot_returns_five_results(monkeypatch):
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", p.REAL_JOB_SNAPSHOT_PATH)
    results = p.run_full_codesign_pipeline(p.DEFAULT_CANDIDATE_PROFILE, top_n=20, data_source=p.REAL_JOB_SNAPSHOT)
    assert len(results) == 5
    assert set(results["candidate_id"]) == set(range(5))
    assert results["job_url"].notna().all()
    assert results["job_url"].astype(str).str.startswith("https://to.indeed.com/").all()
    assert results["title"].tolist() == [
        "Healthcare Data Scientist",
        "Business Intelligence Analyst",
        "Data Scientist I",
        "Data Quality Analyst (Translational Research)",
        "AI Solutions Data Analyst",
    ]
    assert results["match_percent"].tolist() == [100.0, 95.2, 84.2, 79.7, 71.8]
    assert results["evaluation_coverage_percent"].tolist() == [20.0, 50.0, 20.0, 20.0, 20.0]
    assert results["recommendation"].tolist() == ["Consider"] * 5


def test_real_job_missing_location_does_not_default_to_remote(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {
        "jobs": [
            {
                "job_id": "R001",
                "title": "Public Analyst",
                "company": "Example Company",
                "required_skills": ["sql"],
                "description": "Concise public-posting excerpt requiring SQL."
            }
        ]
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)

    jobs = p.load_and_preprocess_dataset(data_source=p.REAL_JOB_SNAPSHOT)

    assert len(jobs) == 1
    assert pd.isna(jobs.iloc[0]["location"])


def test_real_job_missing_work_mode_and_experience_stay_unknown(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {
        "jobs": [
            {
                "job_id": "R001",
                "title": "Public Analyst",
                "company": "Example Company",
                "location": "Kansas City, MO",
                "required_skills": ["sql"],
                "description": "Concise public-posting excerpt requiring SQL."
            }
        ]
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)

    jobs = p.load_and_preprocess_dataset(data_source=p.REAL_JOB_SNAPSHOT)
    extraction = p.fallback_extract_job_requirements(jobs.iloc[0])

    assert pd.isna(jobs.iloc[0]["work_mode"])
    assert pd.isna(jobs.iloc[0]["min_experience"])
    assert extraction["experience_years"] is None


def test_real_job_missing_title_or_company_is_not_fabricated(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {
        "jobs": [
            {
                "job_id": "R001",
                "title": "",
                "company": "Example Company",
                "description": "Concise excerpt."
            },
            {
                "job_id": "R002",
                "title": "Public Analyst",
                "company": None,
                "description": "Concise excerpt."
            },
            {
                "job_id": "R003",
                "title": "Public Data Analyst",
                "company": "Example Company",
                "description": "Concise excerpt."
            }
        ]
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)

    jobs = p.load_and_preprocess_dataset(data_source=p.REAL_JOB_SNAPSHOT)

    assert len(jobs) == 1
    assert jobs.iloc[0]["job_id"] == "R003"
    assert "Data Scientist" not in jobs["title"].tolist()
    assert "Tech Corp" not in jobs["company"].tolist()


def test_real_job_missing_title_or_company_columns_fail_clearly(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "real_jobs.json"
    snapshot = {"jobs": [{"job_id": "R001", "description": "Concise excerpt."}]}
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(p, "REAL_JOB_SNAPSHOT_PATH", snapshot_path)

    with pytest.raises(ValueError, match="Real Job Snapshot records require fields"):
        p.load_and_preprocess_dataset(data_source=p.REAL_JOB_SNAPSHOT)
