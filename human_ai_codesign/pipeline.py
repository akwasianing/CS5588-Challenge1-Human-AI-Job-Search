"""
Human–AI Co-Designed Job Matching Pipeline
Validated implementation based on Challenge1_Public_Job_Data.ipynb
"""

import json
import re
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Candidate Matching Profile (Validated Default)
DEFAULT_CANDIDATE_PROFILE = {
    "education": [
        "MS Data Science and Analytics",
        "BSc Medical Laboratory Technology"
    ],
    "skills": [
        "Python", "R", "SQL", "PySpark", "Tableau", "Power BI", "Excel", "KNIME",
        "Machine Learning", "Predictive Modeling", "Statistical Analysis",
        "Data Visualization", "Data Analysis", "Natural Language Processing",
        "LangChain", "ChromaDB", "Streamlit"
    ],
    "experience_years": {
        "data_analytics": 2,
        "biomedical_research": 8,
        "healthcare_biomedical": 8
    },
    "preferred_roles": [
        "Data Scientist", "Data Analyst", "Healthcare Data Scientist",
        "Healthcare Data Analyst", "Biomedical Informatics", "Clinical Data", "Business Analytics"
    ],
    "preferred_locations": ["Kansas City, MO", "Phoenix, AZ", "Remote"],
    "preferred_work_arrangements": ["Remote", "Hybrid"],
    "preferred_job_types": ["Full-time"]
}

MATCHING_WEIGHTS = {
    "skills": 0.30,
    "experience_projects": 0.25,
    "education": 0.20,
    "role_alignment": 0.10,
    "location_work_arrangement": 0.10,
    "salary_job_type": 0.05
}

EVALUATION_DATASET = "Evaluation Dataset"
REAL_JOB_SNAPSHOT = "Real Job Snapshot"
REAL_JOB_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "real_job_snapshot.json"

FALLBACK_JOBS_DATA = [
    {"job_id":"J001","title":"Junior Data Scientist","company":"HealthAI Labs","location":"Kansas City, MO","work_mode":"Hybrid","min_experience":1,"required_skills":["python","sql","pandas","machine learning"],"preferred_skills":["healthcare","scikit-learn","git"],"description":"Build predictive models and analytics pipelines for healthcare data using Python, SQL, pandas, and machine learning. U.S. Citizenship or Work Authorization required."},
    {"job_id":"J002","title":"Data Analyst","company":"Metro Analytics","location":"Kansas City, MO","work_mode":"On-site","min_experience":1,"required_skills":["sql","excel","data visualization"],"preferred_skills":["python","tableau"],"description":"Analyze business data, create SQL reports, dashboards, and visualizations, and communicate findings to stakeholders."},
    {"job_id":"J003","title":"Machine Learning Engineer","company":"VectorWorks AI","location":"Remote","work_mode":"Remote","min_experience":3,"required_skills":["python","pytorch","docker","machine learning"],"preferred_skills":["aws","mlops","git"],"description":"Develop production machine learning systems, model training pipelines, APIs, containers, and cloud deployments. Active Secret Clearance required."},
    {"job_id":"J004","title":"Research Data Scientist","company":"BioDiscovery Institute","location":"St. Louis, MO","work_mode":"Hybrid","min_experience":2,"required_skills":["python","statistics","machine learning"],"preferred_skills":["healthcare","nlp","pandas"],"description":"Apply statistical learning and AI methods to biomedical research datasets and collaborate with domain scientists."},
    {"job_id":"J005","title":"Business Intelligence Analyst","company":"Civic Data Group","location":"Remote","work_mode":"Remote","min_experience":2,"required_skills":["sql","power bi","data visualization"],"preferred_skills":["python","data modeling"],"description":"Build dashboards, semantic models, SQL transformations, and KPI reporting for distributed business teams."},
    {"job_id":"J006","title":"Healthcare Data Analyst","company":"CareMetrics","location":"Remote","work_mode":"Remote","min_experience":1,"required_skills":["sql","python","data visualization"],"preferred_skills":["healthcare","pandas"],"description":"Analyze healthcare quality data with SQL and Python, prepare visual reports, and support clinical analytics teams."},
    {"job_id":"J007","title":"Data Engineering Associate","company":"CloudPipe","location":"Kansas City, MO","work_mode":"Hybrid","min_experience":2,"required_skills":["python","sql","etl"],"preferred_skills":["spark","aws","git"],"description":"Create ETL workflows, data quality checks, SQL transformations, and cloud-oriented data pipelines."},
    {"job_id":"J008","title":"NLP Research Assistant","company":"University AI Lab","location":"Kansas City, MO","work_mode":"On-site","min_experience":0,"required_skills":["python","machine learning"],"preferred_skills":["nlp","hugging face","research"],"description":"Support experiments in natural language processing, machine learning, Hugging Face models, and research evaluation."}
]

def normalize_work_mode(value):
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if "remote" in text:
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    if "on-site" in text or "onsite" in text or "on site" in text:
        return "On-site"
    return str(value).strip()

def is_remote_or_hybrid_work_mode(value):
    return normalize_work_mode(value) in {"Remote", "Hybrid"}

def get_seniority(title):
    title = str(title).lower()
    if any(term in title for term in ["director", "head of", "vp", "vice president", "chief"]):
        return "Director/Executive"
    elif any(term in title for term in ["lead", "principal"]):
        return "Lead/Principal"
    elif any(term in title for term in ["senior", "sr"]):
        return "Senior"
    elif any(term in title for term in ["intern", "trainee"]):
        return "Intern"
    elif any(term in title for term in ["junior", "jr", "assistant", "associate", "entry"]):
        return "Entry/Junior"
    else:
        return "Mid-level"

def load_real_job_snapshot(path=None):
    snapshot_path = Path(path) if path is not None else REAL_JOB_SNAPSHOT_PATH
    if not snapshot_path.exists():
        return []
    with open(snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("jobs", [])
    if isinstance(data, list):
        return data
    return []

def load_and_preprocess_dataset(data_source=EVALUATION_DATASET):
    df = None
    is_real_snapshot = data_source == REAL_JOB_SNAPSHOT
    if data_source == REAL_JOB_SNAPSHOT:
        df = pd.DataFrame(load_real_job_snapshot())
    elif HAS_DATASETS:
        try:
            dataset = load_dataset("keerthanshetty/r-datascientist-jobs")
            df = dataset["train"].to_pandas()
        except Exception:
            df = pd.DataFrame(FALLBACK_JOBS_DATA)
    else:
        df = pd.DataFrame(FALLBACK_JOBS_DATA)
        
    jobs = df.copy()
    
    def clean_skills(val):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, list):
            return [str(s).strip().lower() for s in val if s]
        if isinstance(val, str) and val.strip():
            return [str(s).strip().lower() for s in val.split(",") if s.strip()]
        return []

    if is_real_snapshot and not jobs.empty:
        missing_required = []
        for col in ["title", "company"]:
            if col not in jobs.columns:
                missing_required.append(col)
        if missing_required:
            raise ValueError(f"Real Job Snapshot records require fields: {missing_required}")
        valid_required = jobs["title"].notna() & jobs["company"].notna()
        valid_required &= jobs["title"].astype(str).str.strip().ne("")
        valid_required &= jobs["company"].astype(str).str.strip().ne("")
        jobs = jobs[valid_required].reset_index(drop=True)

    if "required_skills" in jobs.columns:
        jobs["required_skills_list"] = jobs["required_skills"].apply(clean_skills)
    elif "skills" in jobs.columns:
        jobs["required_skills_list"] = jobs["skills"].apply(clean_skills)
    else:
        jobs["required_skills_list"] = [[] for _ in range(len(jobs))]

    if "title" not in jobs.columns:
        jobs["title"] = "Data Scientist"
    if "company" not in jobs.columns:
        jobs["company"] = "Tech Corp"
    if "location" not in jobs.columns:
        jobs["location"] = None if is_real_snapshot else "Remote"
    if "description" not in jobs.columns:
        jobs["description"] = jobs["title"]
    if "work_mode" not in jobs.columns:
        jobs["work_mode"] = None
    if "min_experience" not in jobs.columns:
        jobs["min_experience"] = None
    if "preferred_skills" in jobs.columns:
        jobs["preferred_skills_list"] = jobs["preferred_skills"].apply(clean_skills)
    else:
        jobs["preferred_skills_list"] = [[] for _ in range(len(jobs))]
    for optional_col in ["job_url", "source", "date_retrieved"]:
        if optional_col not in jobs.columns:
            jobs[optional_col] = None

    if jobs.empty:
        jobs["seniority_level"] = []
        jobs["skills_text"] = []
        jobs["search_text"] = []
        return jobs

    jobs["seniority_level"] = jobs["title"].apply(get_seniority)
    jobs["skills_text"] = jobs["required_skills_list"].apply(lambda x: " ".join(x))
    jobs["search_text"] = (
        jobs["title"].fillna("") + " " +
        jobs["skills_text"] + " " +
        jobs["description"].fillna("")
    )
    
    jobs = jobs.drop_duplicates(subset=["title", "company", "description"]).reset_index(drop=True)
    return jobs

def tokenize_text(text):
    return re.findall(r"\w+", str(text).lower())

class RobustEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if HAS_SBERT:
            try:
                self.sbert = SentenceTransformer(model_name)
                self.mode = "sbert"
            except Exception:
                self.mode = "tfidf"
        else:
            self.mode = "tfidf"

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if self.mode == "sbert":
            return self.sbert.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
        else:
            vec = TfidfVectorizer(token_pattern=r"\w+")
            matrix = vec.fit_transform(texts).toarray()
            if normalize_embeddings and matrix.shape[0] > 0:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                matrix = matrix / norms
            return matrix

class BM25HybridRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = RobustEmbedder(model_name)

    def retrieve(self, candidate_profile, jobs_df, top_n=20):
        skills_str = " ".join(candidate_profile["skills"])
        roles_str = " ".join(candidate_profile["preferred_roles"])
        query_str = f"{skills_str} {roles_str}"
        tokenized_query = tokenize_text(query_str)
        
        # BM25 Lexical Scoring
        tokenized_corpus = [tokenize_text(txt) for txt in jobs_df["search_text"]]
        if HAS_BM25:
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)
        else:
            bm25_scores = np.array([sum(t in txt for t in tokenized_query) for txt in tokenized_corpus], dtype=float)
        
        # Semantic Embedding Scoring
        job_texts = jobs_df["search_text"].tolist()
        all_texts = [query_str] + job_texts
        all_embs = self.embedder.encode(all_texts, normalize_embeddings=True)
        candidate_emb = all_embs[0:1]
        job_embs = all_embs[1:]
        semantic_scores = cosine_similarity(candidate_emb, job_embs).flatten()
        
        # MinMax Normalization & 50/50 Hybrid
        scaler = MinMaxScaler()
        bm25_norm = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
        semantic_norm = scaler.fit_transform(np.array(semantic_scores).reshape(-1, 1)).flatten()
        
        hybrid_scores = 0.50 * bm25_norm + 0.50 * semantic_norm
        
        res_df = jobs_df.copy()
        res_df["bm25_score"] = bm25_scores
        res_df["bm25_norm"] = bm25_norm
        res_df["semantic_score"] = semantic_scores
        res_df["semantic_norm"] = semantic_norm
        res_df["hybrid_score"] = hybrid_scores
        
        top_candidates = res_df.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)
        return top_candidates

def clean_llm_json(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()

def validate_extraction(data):
    if not isinstance(data, dict):
        return {"is_valid": False, "warnings": ["Extraction output is not a JSON object"]}
    expected_keys = [
        "required_skills", "preferred_skills", "education_requirements",
        "experience_years", "seniority_level", "location", "work_arrangement",
        "salary", "job_type", "eligibility_requirements"
    ]
    missing_keys = [key for key in expected_keys if key not in data]
    warnings = []
    if missing_keys:
        warnings.append(f"Missing keys: {missing_keys}")
    return {"is_valid": len(warnings) == 0, "warnings": warnings}

def repair_missing_evidence(data, job_description):
    repaired_data = json.loads(json.dumps(data))
    desc_lower = str(job_description).lower()
    
    def check_items(items):
        repaired = []
        for item in items:
            text = item.get("text", "")
            quote = item.get("quote", "")
            if quote and str(quote).lower() in desc_lower:
                repaired.append(item)
            elif text and str(text).lower() in desc_lower:
                repaired.append({"text": text, "quote": text})
            else:
                repaired.append({"text": text, "quote": "Extracted from job text"})
        return repaired

    for key in ["required_skills", "preferred_skills", "education_requirements", "eligibility_requirements"]:
        if key in repaired_data and isinstance(repaired_data[key], list):
            repaired_data[key] = check_items(repaired_data[key])
            
    return repaired_data

def fallback_extract_job_requirements(row):
    desc = str(row["description"])
    skills = row.get("required_skills_list", [])
    
    req_skills = [{"text": s, "quote": s} for s in skills]
    pref_skills = []
    
    desc_lower = desc.lower()
    edu_reqs = []
    if re.search(r"(?<!\w)(phd|ph\.d\.?|doctorate)(?!\w)", desc_lower):
        edu_reqs.append({"text": "PhD in quantitative discipline", "quote": "PhD"})
    elif re.search(r"(?<!\w)(master|masters|master's|ms|m\.s\.?)(?!\w)", desc_lower):
        edu_reqs.append({"text": "Master's degree in quantitative discipline", "quote": "Master's degree"})
    elif re.search(r"(?<!\w)(bachelor|bachelors|bachelor's|bs|b\.s\.?)(?!\w)", desc_lower):
        edu_reqs.append({"text": "Bachelor's degree in quantitative discipline", "quote": "Bachelor's degree"})

    min_experience = row.get("min_experience", None)
    exp_years = None
    if pd.notna(min_experience):
        try:
            exp_years = int(float(min_experience))
        except (TypeError, ValueError):
            exp_years = None
    if exp_years is None:
        exp_match = re.search(r"(\d+)\+?\s*years?", desc, re.IGNORECASE)
        exp_years = int(exp_match.group(1)) if exp_match else None

    elig_reqs = []
    if any(k in desc_lower for k in ["clearance", "public trust", "secret"]):
        elig_reqs.append({"text": "Security clearance required", "quote": "clearance"})
    if any(k in desc_lower for k in ["citizen", "citizenship"]):
        elig_reqs.append({"text": "U.S. Citizenship required", "quote": "citizen"})
    if any(k in desc_lower for k in ["sponsorship", "visa"]):
        elig_reqs.append({"text": "Visa sponsorship information", "quote": "sponsorship"})

    loc = row.get("location", "Not specified")
    work_arr = normalize_work_mode(row.get("work_mode", None))
    if work_arr is None:
        work_arr = "Remote" if "remote" in desc_lower else ("Hybrid" if "hybrid" in desc_lower else "On-site")
    
    data = {
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "education_requirements": edu_reqs,
        "experience_years": exp_years,
        "seniority_level": row.get("seniority_level", "Mid-level"),
        "location": loc,
        "work_arrangement": work_arr,
        "salary": None,
        "job_type": None,
        "eligibility_requirements": elig_reqs
    }
    return repair_missing_evidence(data, desc)

class CoDesignScorer:
    def __init__(self, embedder=None):
        if embedder is None:
            self.embedder = RobustEmbedder()
        else:
            self.embedder = embedder

    def semantic_skill_match(self, req_skill, candidate_skills):
        all_texts = [req_skill] + candidate_skills
        all_embs = self.embedder.encode(all_texts, normalize_embeddings=True)
        req_emb = all_embs[0:1]
        cand_embs = all_embs[1:]
        sims = cosine_similarity(req_emb, cand_embs).flatten()
        best_score = float(np.max(sims)) if len(sims) > 0 else 0.0
        return best_score if best_score >= 0.70 else 0.0

    def score_skills(self, req_skills_list, candidate_skills, mode=None):
        """Score skill match.

        If the job posting provides no explicit required skills (empty list),
        the function returns ``np.nan`` so the factor is excluded from final-score renormalisation.
        When required skills are present, the original exact/semantic matching logic is retained.
        The optional ``mode`` argument is ignored (kept for backward compatibility).
        """
        if not req_skills_list:
            return np.nan
        cand_lower = {s.lower() for s in candidate_skills}
        matched_scores = []
        for req_skill in req_skills_list:
            s_lower = str(req_skill).lower()
            if s_lower in cand_lower or any(s_lower in c for c in cand_lower):
                matched_scores.append(1.0)
            else:
                sem_s = self.semantic_skill_match(s_lower, list(cand_lower))
                if sem_s > 0.70:
                    matched_scores.append(sem_s)
                else:
                    matched_scores.append(0.0)
        score = sum(matched_scores) / len(req_skills_list)
        return float(score)



# The function now returns np.nan so that the factor is excluded from final-score renormalisation. When required skills are present the original exact/semantic matching logic is retained.


    def get_degree_rank(self, text):
        text_lower = str(text).lower()
        if any(k in text_lower for k in ["phd", "ph.d", "doctorate"]):
            return 3
        elif any(k in text_lower for k in ["master", "ms", "m.s.", "msc"]):
            return 2
        elif any(k in text_lower for k in ["bachelor", "bs", "b.s.", "bsc"]):
            return 1
        return 0

    def score_education(self, edu_reqs, candidate_edu, mode=None):
        cand_highest_rank = max([self.get_degree_rank(e) for e in candidate_edu] + [1])
        if not edu_reqs:
            return np.nan
        req_ranks = [self.get_degree_rank(item.get("text", "")) for item in edu_reqs if isinstance(item, dict)]
        req_rank = max(req_ranks) if req_ranks else 1
        
        if cand_highest_rank >= req_rank:
            return 1.0
        elif cand_highest_rank == req_rank - 1:
            return 0.75
        else:
            return 0.50

    def score_experience(self, req_years, seniority, candidate_years_dict, job_desc):
        # Years of experience score: only if a numeric requirement is provided
        if req_years is not None:
            req_y = req_years
            cand_y = candidate_years_dict.get("data_analytics", 0)
            y_score = 1.0 if cand_y >= req_y else cand_y / max(req_y, 1)
        else:
            y_score = np.nan
        return float(y_score) if not np.isnan(y_score) else np.nan

    def score_role_alignment(self, job_title, seniority, preferred_roles):
        all_texts = [job_title] + preferred_roles
        all_embs = self.embedder.encode(all_texts, normalize_embeddings=True)
        job_emb = all_embs[0:1]
        pref_embs = all_embs[1:]
        sims = cosine_similarity(job_emb, pref_embs).flatten()
        best_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
        
        if seniority == "Director/Executive":
            role_score = best_sim * 0.50
        elif seniority == "Senior":
            role_score = best_sim * 0.85
        else:
            role_score = best_sim
        return float(np.clip(role_score, 0.0, 1.0))

    def score_location_work(self, extraction, candidate_locs, candidate_works):
        work_arr = normalize_work_mode(extraction.get("work_arrangement", "Remote"))
        loc = extraction.get("location", "Remote")
        
        work_match = 1.0 if work_arr in candidate_works or "remote" in str(work_arr).lower() else 0.50
        loc_match = 1.0 if any(pl.lower() in str(loc).lower() for pl in candidate_locs) or "remote" in str(loc).lower() else 0.50
        
        return float(0.50 * work_match + 0.50 * loc_match)

    def score_salary_job_type(self, extraction):
        # Salary is ignored as candidate has no salary preference
        job_type = extraction.get("job_type")
        # If no job_type information, return nan
        if job_type is None:
            return np.nan
        scores = []
        if job_type and "full" in str(job_type).lower():
            scores.append(1.0)
        # Salary is deliberately not scored
        return float(np.mean(scores)) if scores else np.nan

def validate_hard_eligibility(job_title, extraction):
    requirements = extraction.get("eligibility_requirements", [])
    has_clearance = False
    has_citizen_auth = False
    
    for req in requirements:
        text = str(req.get("text", "")).lower()
        if any(k in text for k in ["clearance", "public trust", "secret"]):
            has_clearance = True
        elif any(k in text for k in ["citizen", "citizenship", "work authorization", "sponsorship", "visa"]):
            has_citizen_auth = True
            
    if has_clearance:
        return "Hard eligibility requirement present"
    elif has_citizen_auth:
        return "Human review required"
    else:
        return "No hard eligibility requirement detected"

def create_application_recommendation(row):
    score = row["match_percent"]
    coverage = row["evaluation_coverage"]
    eligibility = row["eligibility_status"]

    if eligibility == "Hard eligibility requirement present":
        return "Review Eligibility First"
    if eligibility == "Human review required":
        if score is not None and score >= 75:
            return "Consider - Check Eligibility"
        else:
            return "Low Priority - Check Eligibility"

    if score is None:
        return "Unable to evaluate"
    elif score >= 80 and coverage in ["High", "Moderate"]:
        return "Apply"
    elif score >= 65:
        return "Consider"
    else:
        return "Low Priority"

def apply_human_decision(ai_recommendation, human_decision):
    if human_decision in ["Accept AI", "Accept Recommendation"]:
        return ai_recommendation
    elif human_decision == "Apply":
        return "Apply"
    elif human_decision == "Consider":
        return "Consider"
    elif human_decision == "Reject / Skip":
        return "Excluded by Human"
    elif human_decision == "Needs Manual Review":
        return "Needs Manual Review"
    return ai_recommendation

RESULT_COLUMNS = [
    "candidate_id", "title", "company", "location", "work_mode", "min_experience",
    "required_skills", "preferred_skills", "job_url", "source", "date_retrieved",
    "hybrid_score", "bm25_score", "semantic_score", "final_match_score",
    "match_percent", "match_tier", "evaluation_coverage_percent",
    "evaluation_coverage", "skills_score", "experience_projects_score",
    "education_score", "role_alignment_score", "location_work_score",
    "salary_job_type_score", "unknown_factors", "eligibility_status",
    "description", "matched_skills", "missing_skills", "recommendation"
]

def run_full_codesign_pipeline(candidate_profile=None, top_n=20, data_source=EVALUATION_DATASET):
    if candidate_profile is None:
        candidate_profile = DEFAULT_CANDIDATE_PROFILE
        
    jobs_df = load_and_preprocess_dataset(data_source=data_source)
    if jobs_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    retriever = BM25HybridRetriever()
    top_candidates = retriever.retrieve(candidate_profile, jobs_df, top_n=top_n)
    scorer = CoDesignScorer(retriever.embedder)
    
    final_results = []
    for i, row in top_candidates.iterrows():
        extraction = fallback_extract_job_requirements(row)
        eligibility_status = validate_hard_eligibility(row["title"], extraction)
        
        skill_s = scorer.score_skills(row["required_skills_list"], candidate_profile["skills"])
        edu_s = scorer.score_education(extraction["education_requirements"], candidate_profile["education"])
        exp_s = scorer.score_experience(extraction["experience_years"], row["seniority_level"], candidate_profile["experience_years"], row["description"])
        role_s = scorer.score_role_alignment(row["title"], row["seniority_level"], candidate_profile["preferred_roles"])
        loc_s = scorer.score_location_work(extraction, candidate_profile["preferred_locations"], candidate_profile["preferred_work_arrangements"])
        sal_s = scorer.score_salary_job_type(extraction)

        candidate_skills_lower = {str(s).lower() for s in candidate_profile["skills"]}
        matched_skills = []
        missing_skills = []
        for req_skill in row["required_skills_list"]:
            req_skill_lower = str(req_skill).lower()
            if req_skill_lower in candidate_skills_lower or any(req_skill_lower in c for c in candidate_skills_lower):
                matched_skills.append(req_skill)
            else:
                missing_skills.append(req_skill)
        
        component_scores = {
            "skills": skill_s,
            "experience_projects": exp_s,
            "education": edu_s,
            "role_alignment": role_s,
            "location_work_arrangement": loc_s,
            "salary_job_type": sal_s
        }
        
        weighted_sum = 0.0
        available_weight = 0.0
        unknown_factors = []
        
        for factor, weight in MATCHING_WEIGHTS.items():
            s = component_scores[factor]
            if pd.isna(s):
                unknown_factors.append(factor)
            else:
                weighted_sum += s * weight
                available_weight += weight
                
        final_score = weighted_sum / available_weight if available_weight > 0 else None
        match_percent = round(final_score * 100, 1) if final_score is not None else None
        
        if final_score is None:
            tier = "Unable to evaluate"
        elif final_score >= 0.80:
            tier = "Strong match"
        elif final_score >= 0.65:
            tier = "Good match"
        elif final_score >= 0.50:
            tier = "Moderate match"
        else:
            tier = "Low match"
            
        coverage_percent = round(available_weight * 100, 1)
        coverage_level = "High" if coverage_percent >= 85 else ("Moderate" if coverage_percent >= 70 else "Limited")
        
        res = {
            "candidate_id": i,
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "work_mode": row.get("work_mode", None),
            "min_experience": row.get("min_experience", None),
            "required_skills": row.get("required_skills_list", []),
            "preferred_skills": row.get("preferred_skills_list", []),
            "job_url": row.get("job_url", row.get("apply_url", row.get("application_url", None))),
            "source": row.get("source", None),
            "date_retrieved": row.get("date_retrieved", None),
            "hybrid_score": row["hybrid_score"],
            "bm25_score": row["bm25_score"],
            "semantic_score": row["semantic_score"],
            "final_match_score": final_score,
            "match_percent": match_percent,
            "match_tier": tier,
            "evaluation_coverage_percent": coverage_percent,
            "evaluation_coverage": coverage_level,
            "skills_score": skill_s,
            "experience_projects_score": exp_s,
            "education_score": edu_s,
            "role_alignment_score": role_s,
            "location_work_score": loc_s,
            "salary_job_type_score": sal_s,
            "unknown_factors": unknown_factors,
            "eligibility_status": eligibility_status,
            "description": row["description"],
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }
        res["recommendation"] = create_application_recommendation(res)
        final_results.append(res)
        
    res_df = pd.DataFrame(final_results).sort_values("final_match_score", ascending=False).reset_index(drop=True)
    return res_df
