"""
Unit & Parity Test Suite for Human–AI Co-Designed Job Search Engine
Verifies pipeline mechanics, scoring boundedness, eligibility classification, weight renormalization, and human overrides.
"""

import unittest
import numpy as np
import pandas as pd
from human_ai_codesign.pipeline import (
    MATCHING_WEIGHTS,
    DEFAULT_CANDIDATE_PROFILE,
    BM25HybridRetriever,
    fallback_extract_job_requirements,
    get_seniority,
    is_remote_or_hybrid_work_mode,
    validate_hard_eligibility,
    create_application_recommendation,
    apply_human_decision,
    CoDesignScorer
)

class TestPipelineMechanics(unittest.TestCase):

    def test_six_factor_weights_sum_to_one(self):
        self.assertEqual(list(MATCHING_WEIGHTS.values()), [0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
        total_weight = sum(MATCHING_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.00, places=4, msg="Matching base weights must sum to 1.00 (100%)")
        self.assertEqual(len(MATCHING_WEIGHTS), 6, msg="Pipeline must enforce exactly 6 co-designed factors")

    def test_unknown_factor_renormalization(self):
        # Simulate unstated salary_job_type
        component_scores = {
            "skills": 0.90,
            "experience_projects": 1.00,
            "education": 0.75,
            "role_alignment": 1.00,
            "location_work_arrangement": 1.00,
            "salary_job_type": np.nan  # Unknown factor
        }
        
        weighted_sum = 0.0
        available_weight = 0.0
        for factor, weight in MATCHING_WEIGHTS.items():
            s = component_scores[factor]
            if not pd.isna(s):
                weighted_sum += s * weight
                available_weight += weight
                
        self.assertAlmostEqual(available_weight, 0.95, places=4, msg="Available weight must exclude missing factor (0.95)")
        final_score = weighted_sum / available_weight
        self.assertGreater(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)
        
        coverage_percent = round(available_weight * 100, 1)
        self.assertEqual(coverage_percent, 95.0)

    def test_eligibility_classification_states(self):
        # 1. Hard clearance requirement
        ext_clearance = {"eligibility_requirements": [{"text": "Active Top Secret Clearance required", "quote": "Clearance"}]}
        res1 = validate_hard_eligibility("Data Scientist", ext_clearance)
        self.assertEqual(res1, "Hard eligibility requirement present")

        # 2. Citizenship / Visa sponsorship requirement without candidate data
        ext_citizen = {"eligibility_requirements": [{"text": "Must be a US Citizen or Permanent Resident", "quote": "Citizen"}]}
        res2 = validate_hard_eligibility("Data Scientist", ext_citizen)
        self.assertEqual(res2, "Human review required")

        # 3. HIPAA / privacy requirement (Not hard eligibility)
        ext_hipaa = {"eligibility_requirements": [{"text": "Knowledge of HIPAA compliance", "quote": "HIPAA"}]}
        res3 = validate_hard_eligibility("Data Scientist", ext_hipaa)
        self.assertEqual(res3, "No hard eligibility requirement detected")

    def test_eligibility_outside_weighted_score(self):
        row_clearance = {
            "match_percent": 92.5,
            "evaluation_coverage": "High",
            "eligibility_status": "Hard eligibility requirement present"
        }
        rec = create_application_recommendation(row_clearance)
        self.assertEqual(rec, "Review Eligibility First", msg="Clearance requirement must override AI recommendation label")
        # Verify numerical match_percent remains 92.5 (un-altered)
        self.assertEqual(row_clearance["match_percent"], 92.5)

    def test_human_decision_overrides(self):
        ai_rec = "Apply"
        
        # Test accept recommendation labels
        self.assertEqual(apply_human_decision(ai_rec, "Accept AI"), "Apply")
        self.assertEqual(apply_human_decision(ai_rec, "Accept Recommendation"), "Apply")
        # Test Human Reject
        self.assertEqual(apply_human_decision(ai_rec, "Reject / Skip"), "Excluded by Human")
        # Test Manual Review
        self.assertEqual(apply_human_decision(ai_rec, "Needs Manual Review"), "Needs Manual Review")

    def test_seniority_extraction(self):
        self.assertEqual(get_seniority("Director of AI"), "Director/Executive")
        self.assertEqual(get_seniority("Senior Data Scientist"), "Senior")
        self.assertEqual(get_seniority("Junior Analyst"), "Entry/Junior")
        self.assertEqual(get_seniority("Data Science Intern"), "Intern")
        self.assertEqual(get_seniority("Data Scientist"), "Mid-level")

    def test_explicit_work_mode_is_respected(self):
        remote_row = pd.Series({
            "description": "Analyze healthcare quality data.",
            "required_skills_list": [],
            "location": "Remote",
            "work_mode": "Remote",
            "min_experience": None,
            "seniority_level": "Mid-level"
        })
        hybrid_row = pd.Series({
            "description": "Analyze business data.",
            "required_skills_list": [],
            "location": "Kansas City, MO",
            "work_mode": "Hybrid",
            "min_experience": None,
            "seniority_level": "Mid-level"
        })

        self.assertEqual(fallback_extract_job_requirements(remote_row)["work_arrangement"], "Remote")
        self.assertEqual(fallback_extract_job_requirements(hybrid_row)["work_arrangement"], "Hybrid")

    def test_min_experience_precedes_description_regex(self):
        row = pd.Series({
            "description": "This description mentions 7 years in passing.",
            "required_skills_list": [],
            "location": "Remote",
            "work_mode": "Remote",
            "min_experience": 2,
            "seniority_level": "Mid-level"
        })
        extraction = fallback_extract_job_requirements(row)
        self.assertEqual(extraction["experience_years"], 2)

    def test_description_regex_used_when_min_experience_missing(self):
        row = pd.Series({
            "description": "Requires 4 years of analytics experience.",
            "required_skills_list": [],
            "location": "Remote",
            "work_mode": None,
            "min_experience": None,
            "seniority_level": "Mid-level"
        })
        extraction = fallback_extract_job_requirements(row)
        self.assertEqual(extraction["experience_years"], 4)

    def test_no_education_requirement_stays_missing(self):
        row = pd.Series({
            "description": "Analyze data for business teams and production systems.",
            "required_skills_list": [],
            "location": "Remote",
            "work_mode": "Remote",
            "min_experience": None,
            "seniority_level": "Mid-level"
        })
        extraction = fallback_extract_job_requirements(row)
        self.assertEqual(extraction["education_requirements"], [])

    def test_remote_hybrid_filter_predicate(self):
        self.assertTrue(is_remote_or_hybrid_work_mode("Remote"))
        self.assertTrue(is_remote_or_hybrid_work_mode("Hybrid"))
        self.assertFalse(is_remote_or_hybrid_work_mode("On-site"))
        self.assertFalse(is_remote_or_hybrid_work_mode("Onsite"))

    def test_retrieval_formula_remains_50_50(self):
        source_names = BM25HybridRetriever.retrieve.__code__.co_consts
        self.assertIn(0.50, source_names)

if __name__ == "__main__":
    unittest.main()
