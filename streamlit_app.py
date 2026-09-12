import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from human_ai_codesign.pipeline import (
    DEFAULT_CANDIDATE_PROFILE,
    EVALUATION_DATASET,
    MATCHING_WEIGHTS,
    REAL_JOB_SNAPSHOT,
    run_full_codesign_pipeline,
    apply_human_decision,
    is_remote_or_hybrid_work_mode
)
from human_ai_codesign.ui_helpers import (
    ACCEPT_RECOMMENDATION,
    APPLICATION_LINK_UNAVAILABLE,
    build_match_evidence_rows,
    build_stage2_example,
    coverage_aware_tier_label,
    decision_key,
    format_component_score,
    get_human_decision,
    job_url_or_none,
    match_score_label,
    persist_human_decisions
)

st.set_page_config(page_title="AI-Guided Job Search", page_icon="🔎", layout="wide")

# Initialize Session States
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "human_decisions" not in st.session_state:
    st.session_state.human_decisions = {}
if "comparison" not in st.session_state:
    st.session_state.comparison = pd.DataFrame([
        {
            "Version": "Human baseline",
            "Development time (min)": 45,
            "Tests passed": 4,
            "Top-5 quality (1-5)": 3.5,
            "Human edits": "Baseline written manually using 4 rigid rules",
            "Main failure": "Rigid 0-1 scoring, no semantic matching, penalized missing fields",
            "What we learned": "Human intuition creates clear rules but misses skill transferability"
        },
        {
            "Version": "AI-guided",
            "Development time (min)": 20,
            "Tests passed": 3,
            "Top-5 quality (1-5)": 4.0,
            "Human edits": "Rejected Regex Extractor artifact after review",
            "Main failure": "Regex extractor confused required/preferred, missed education, misinterpreted work mode",
            "What we learned": "Regex alone fails as primary requirement extractor; LLM extraction required"
        },
        {
            "Version": "Human-AI co-designed",
            "Development time (min)": 30,
            "Tests passed": 7,
            "Top-5 quality (1-5)": 4.9,
            "Human edits": "LLM extraction + quote repair + eligibility gate + human decision override",
            "Main failure": "None; unknown factors excluded & eligibility kept outside weighted score",
            "What we learned": "LLM + deterministic evidence repair + human override yields highest accuracy"
        },
    ])

# Sidebar Profile Controls
st.sidebar.header("Candidate Profile")
cand_name = st.sidebar.text_input("Candidate Name", "Obed Aning")
cand_edu = st.sidebar.text_area("Education", "\n".join(DEFAULT_CANDIDATE_PROFILE["education"]), height=80).split("\n")
cand_skills = [x.strip() for x in st.sidebar.text_area("Skills (comma separated)", ", ".join(DEFAULT_CANDIDATE_PROFILE["skills"]), height=120).split(",") if x.strip()]
cand_roles = [x.strip() for x in st.sidebar.text_input("Preferred Roles", ", ".join(DEFAULT_CANDIDATE_PROFILE["preferred_roles"])).split(",") if x.strip()]
cand_locs = [x.strip() for x in st.sidebar.text_input("Preferred Locations", ", ".join(DEFAULT_CANDIDATE_PROFILE["preferred_locations"])).split(",") if x.strip()]

candidate_profile = {
    "education": cand_edu,
    "skills": cand_skills,
    "experience_years": DEFAULT_CANDIDATE_PROFILE["experience_years"],
    "preferred_roles": cand_roles,
    "preferred_locations": cand_locs,
    "preferred_work_arrangements": DEFAULT_CANDIDATE_PROFILE["preferred_work_arrangements"],
    "preferred_job_types": DEFAULT_CANDIDATE_PROFILE["preferred_job_types"]
}

st.sidebar.divider()
st.sidebar.subheader("Validated 6-Factor Weights")
for factor, w in MATCHING_WEIGHTS.items():
    st.sidebar.markdown(f"- **{factor.replace('_', ' ').title()}**: `{w:.0%}`")
st.sidebar.caption("Unknown factors are automatically excluded and active weights renormalized to 100%.")

# Header
st.title("🔎 Human–AI Co-Designed Job Search Application")
st.caption("CS 5588 Challenge 1 • Validated Human–AI Co-Design Pipeline (Public Job Dataset)")

t1, t2, t3 = st.tabs([
    "1. Job Search & Matches",
    "2. How Matches Are Calculated",
    "3. Feedback"
])

@st.cache_data(show_spinner="Loading public dataset and running 2-Stage Hybrid Pipeline...")
def get_pipeline_data(profile, job_source):
    return run_full_codesign_pipeline(profile, top_n=20, data_source=job_source)

with t1:
    st.subheader("Human–AI Co-Designed Job Search & Decision Engine")
    job_source = st.selectbox("Job Source", [EVALUATION_DATASET, REAL_JOB_SNAPSHOT], index=0)
    
    with st.spinner("Executing 2-Stage Hybrid Search & 6-Factor Scoring Engine..."):
        ranked_df = get_pipeline_data(candidate_profile, job_source)
        
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        query = st.text_input("Filter by Title / Keywords", placeholder="e.g., Data Scientist, Healthcare, Analytics")
    with c2:
        loc_filter = st.selectbox("Job Location", ["All"] + sorted(list(set(ranked_df["location"].dropna()))))
    with c3:
        remote_only = st.checkbox("Remote / Hybrid Only", True)
        
    filtered = ranked_df.copy()
    if query:
        filtered = filtered[filtered["title"].str.contains(query, case=False, na=False) | filtered["description"].str.contains(query, case=False, na=False)]
    if loc_filter != "All":
        filtered = filtered[filtered["location"] == loc_filter]
    if remote_only and "work_mode" in filtered.columns:
        filtered = filtered[filtered["work_mode"].apply(is_remote_or_hybrid_work_mode)]
        
    if filtered.empty:
        st.warning("No jobs match your current search filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Candidates Filtered", f"{len(filtered)} / {len(ranked_df)}")
        m2.metric("Top Match on Evaluated Factors", f"{filtered.iloc[0]['match_percent']:.1f}%")
        m3.metric("Evaluation Coverage", f"{filtered.iloc[0]['evaluation_coverage_percent']:.1f}%")
        m4.metric("Hard Eligibility Flags", int(sum(filtered["eligibility_status"] == "Hard eligibility requirement present")))
        st.caption("Match Score reflects only the factors that could be evaluated from the available job information. Evaluation Coverage shows how much of the full 6-factor assessment had sufficient information.")
        
        st.markdown("### Top Match Scores")
        st.bar_chart(filtered.head(8).set_index("title")["match_percent"])
        
        st.markdown("### Top Job Recommendations & Human Review Decision Override")
        st.info("Human-AI Co-Design Rule: System Recommendation can be confirmed or overridden by the Human Reviewer below.")
        
        # Prepare Interactive Decision Editor
        display_cols = ["title", "company", "match_percent", "match_tier", "evaluation_coverage", "evaluation_coverage_percent", "eligibility_status", "recommendation"]
        editor_df = filtered.head(10)[display_cols].copy()
        editor_df.rename(columns={
            "match_percent": "Match %",
            "match_tier": "Tier",
            "evaluation_coverage": "Coverage",
            "eligibility_status": "Eligibility Audit",
            "recommendation": "System Recommendation"
        }, inplace=True)
        editor_df["Tier"] = editor_df.apply(
            lambda row: coverage_aware_tier_label(row["Tier"], row["evaluation_coverage_percent"]),
            axis=1
        )
        editor_df["_decision_key"] = editor_df.apply(lambda row: decision_key(row["title"], row["company"]), axis=1)
        
        editor_df["Human Review Decision"] = editor_df["_decision_key"].apply(
            lambda key: get_human_decision(st.session_state.human_decisions, key)
        )
        
        edited_table = st.data_editor(
            editor_df,
            column_config={
                "Human Review Decision": st.column_config.SelectboxColumn(
                    "Human Review Decision",
                    help="Select human override decision",
                    options=[ACCEPT_RECOMMENDATION, "Apply", "Consider", "Reject / Skip", "Needs Manual Review"],
                    required=True
                ),
                "evaluation_coverage_percent": None,
                "_decision_key": None
            },
            use_container_width=True,
            hide_index=True
        )

        persist_human_decisions(st.session_state.human_decisions, edited_table.to_dict(orient="records"))
        
        # Compute Final Auditable Recommendation
        edited_table["Final Auditable Recommendation"] = edited_table.apply(
            lambda row: apply_human_decision(row["System Recommendation"], row["Human Review Decision"]), axis=1
        )
        
        st.markdown("#### Final Auditable Decision Summary")
        st.dataframe(edited_table[["title", "Match %", "System Recommendation", "Human Review Decision", "Final Auditable Recommendation"]], use_container_width=True, hide_index=True)
        
        st.markdown("### Detailed Explanations for Top Ranked Jobs")
        for idx, row in filtered.head(5).iterrows():
            detail_key = f"{row['title']} @ {row['company']}"
            match_label = match_score_label(row["evaluation_coverage_percent"])
            tier_label = coverage_aware_tier_label(row["match_tier"], row["evaluation_coverage_percent"])
            with st.expander(f"{row['title']} — {row['company']} • {match_label}: {row['match_percent']:.1f}% ({tier_label})"):
                st.markdown(f"**Location / Work Mode**: {row['location']} / {row.get('work_mode', 'Not specified')}")
                st.markdown(f"**Evaluation Coverage**: {row['evaluation_coverage_percent']}% ({row['evaluation_coverage']})")
                st.markdown(f"**Eligibility Status**: `{row['eligibility_status']}`")
                
                a, b, c = st.columns(3)
                with a:
                    st.markdown("**Matched Required Skills**")
                    st.write(", ".join(row["matched_skills"]) if row["matched_skills"] else "None")
                with b:
                    st.markdown("**Missing Skills Gaps**")
                    st.write(", ".join(row["missing_skills"]) if row["missing_skills"] else "None")
                with c:
                    st.markdown("**Component Sub-Scores**")
                    st.write(f"- Skills: `{format_component_score(row['skills_score'])}`")
                    st.write(f"- Experience: `{format_component_score(row['experience_projects_score'])}`")
                    st.write(f"- Education: `{format_component_score(row['education_score'])}`")
                    st.write(f"- Role Alignment: `{format_component_score(row['role_alignment_score'])}`")
                    st.write(f"- Location/Work: `{format_component_score(row['location_work_score'])}`")
                    st.write(f"- Salary/Job Type: `{format_component_score(row['salary_job_type_score'])}`")

                st.markdown("**Evidence Used in This Match**")
                st.dataframe(
                    pd.DataFrame(build_match_evidence_rows(row, candidate_profile)),
                    use_container_width=True,
                    hide_index=True
                )
                    
                st.divider()
                st.markdown("**Job Description Snippet:**")
                st.caption(str(row["description"])[:500] + "...")
                if st.button("View Job Details", key=f"details_{detail_key}"):
                    st.session_state["selected_job_detail"] = detail_key
                if st.session_state.get("selected_job_detail") == detail_key:
                    st.write(f"**Title:** {row['title']}")
                    st.write(f"**Company:** {row['company']}")
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**Work Mode:** {row.get('work_mode', 'Not specified')}")
                    if row.get("source"):
                        st.write(f"**Source:** {row.get('source')}")
                    if row.get("date_retrieved"):
                        st.write(f"**Date Retrieved:** {row.get('date_retrieved')}")
                    st.write(f"**Minimum Experience:** {row.get('min_experience', 'Not specified')}")
                    st.write("**Description:**")
                    st.write(row["description"])
                    st.write("**Required Skills:**")
                    st.write(", ".join(row.get("required_skills", [])) if row.get("required_skills") else "Not specified")
                    st.write("**Preferred Skills:**")
                    st.write(", ".join(row.get("preferred_skills", [])) if row.get("preferred_skills") else "Not specified")
                    st.write(f"**{match_label}:** {row['match_percent']:.1f}%")
                    st.write(f"**Evaluation Coverage:** {row['evaluation_coverage_percent']}% ({row['evaluation_coverage']})")
                    st.write(f"**Eligibility Status:** {row['eligibility_status']}")
                    st.write("**Component Score Explanation:**")
                    st.write(f"- Skills: `{format_component_score(row['skills_score'])}`")
                    st.write(f"- Experience: `{format_component_score(row['experience_projects_score'])}`")
                    st.write(f"- Education: `{format_component_score(row['education_score'])}`")
                    st.write(f"- Role Alignment: `{format_component_score(row['role_alignment_score'])}`")
                    st.write(f"- Location/Work: `{format_component_score(row['location_work_score'])}`")
                    st.write(f"- Salary/Job Type: `{format_component_score(row['salary_job_type_score'])}`")
                    job_url = job_url_or_none(row)
                    if job_url:
                        st.link_button("View Original Posting / Apply", job_url)
                    else:
                        st.info(APPLICATION_LINK_UNAVAILABLE)

with t2:
    st.subheader("Stage 1: Hybrid Job Retrieval — BM25 + Hugging Face Embeddings")
    st.markdown("Combines BM25 lexical search with `sentence-transformers/all-MiniLM-L6-v2` dense embeddings ($50\\% \\text{ BM25} + 50\\% \\text{ Semantic}$).")
    
    if 'ranked_df' in locals() and not ranked_df.empty:
        hybrid_display = ranked_df[["title", "company", "bm25_score", "semantic_score", "hybrid_score"]].head(15).copy()
        hybrid_display["BM25 Raw"] = hybrid_display["bm25_score"].round(2)
        hybrid_display["Semantic Similarity"] = (hybrid_display["semantic_score"] * 100).round(1).astype(str) + "%"
        hybrid_display["Hybrid Score"] = (hybrid_display["hybrid_score"] * 100).round(1).astype(str) + "%"
        st.dataframe(hybrid_display[["title", "company", "BM25 Raw", "Semantic Similarity", "Hybrid Score"]], use_container_width=True, hide_index=True)

    st.subheader("Stage 2: Six-Factor Match Evaluation")
    factor_weights = pd.DataFrame([
        {"Factor": "Skills", "Weight": "30%"},
        {"Factor": "Experience / Projects", "Weight": "25%"},
        {"Factor": "Education", "Weight": "20%"},
        {"Factor": "Role Alignment", "Weight": "10%"},
        {"Factor": "Location / Work Arrangement", "Weight": "10%"},
        {"Factor": "Salary / Job Type", "Weight": "5%"},
    ])
    st.dataframe(factor_weights, use_container_width=True, hide_index=True)
    st.caption("Retrieved Job -> 6-Factor Evaluation -> Unknown Factors Excluded -> Active Weights Renormalized -> Match on Evaluated Factors -> Evaluation Coverage -> Eligibility Check -> System Recommendation")
    st.markdown(
        "A factor is scored only when sufficient information is available. Missing information stays unknown, not zero; "
        "unknown factors are excluded from the weighted score and the remaining active weights are renormalized to 100%. "
        "Evaluation Coverage shows how much of the full 6-factor assessment could actually be evaluated. Eligibility checks "
        "can override the normal score-based recommendation, and the System Recommendation is available for human confirmation "
        "or override in Tab 1."
    )

    example_df = get_pipeline_data(candidate_profile, EVALUATION_DATASET)
    example_match = example_df[
        (example_df["title"] == "Healthcare Data Analyst") &
        (example_df["company"] == "CareMetrics")
    ]
    if not example_match.empty:
        example_row = example_match.iloc[0]
        example = build_stage2_example(example_row, MATCHING_WEIGHTS)
        st.markdown("#### Worked Example: Healthcare Data Analyst - CareMetrics")
        st.dataframe(pd.DataFrame(example["component_rows"]), use_container_width=True, hide_index=True)
        st.write(
            f"Unknown factors: `{', '.join(example['unknown_factors']) if example['unknown_factors'] else 'None'}`"
        )
        st.write(
            f"Available weight / Evaluation Coverage: `{example['available_weight']:.2f}` / "
            f"`{example_row['evaluation_coverage_percent']:.1f}%`"
        )
        st.write(f"Weighted sum: `{example['weighted_sum']:.3f}`")
        st.write(
            f"Renormalized final Match on Evaluated Factors: "
            f"`{example['weighted_sum']:.3f} / {example['available_weight']:.2f} = {example_row['match_percent']:.1f}%`"
        )
        st.write(f"Eligibility status: `{example_row['eligibility_status']}`")
        st.write(f"System Recommendation: `{example_row['recommendation']}`")

with t3:
    st.subheader("Feedback & Refinement Logger")
    if 'ranked_df' in locals() and not ranked_df.empty:
        job_options = [f"{row['title']} @ {row['company']}" for _, row in ranked_df.head(10).iterrows()]
        selected_job = st.selectbox("Select Job for Feedback", job_options)
        action = st.radio("Action", ["save", "reject", "needs review"], horizontal=True)
        note = st.text_input("Notes / Human Correction", placeholder="e.g., clearance requirement needs verification")
        if st.button("Record Feedback", type="primary"):
            st.session_state.feedback.append({"candidate": cand_name, "job": selected_job, "action": action, "note": note})
            st.success("Feedback recorded.")
            
    if st.session_state.feedback:
        fdf = pd.DataFrame(st.session_state.feedback)
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.download_button("Download Feedback CSV", fdf.to_csv(index=False), "feedback_log.csv", "text/csv")

st.divider()
st.caption("CS 5588 Challenge 1 • Validated Teaching Prototype • Keep human review in the loop.")
