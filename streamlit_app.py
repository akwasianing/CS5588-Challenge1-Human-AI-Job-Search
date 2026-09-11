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
    MATCHING_WEIGHTS,
    run_full_codesign_pipeline,
    apply_human_decision
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
st.title("🔎 AI-Guided Job Search Application")
st.caption("CS 5588 Challenge 1 • Validated Human–AI Co-Design Pipeline (Public Job Dataset)")

t1, t2, t3, t4, t5 = st.tabs([
    "1. Search & Co-Design Rank",
    "2. 2-Stage Hybrid Engine",
    "3. Feedback",
    "4. Manual vs AI Experiment",
    "5. Agentic AI + GitHub"
])

@st.cache_data(show_spinner="Loading public dataset and running 2-Stage Hybrid Pipeline...")
def get_pipeline_data(profile):
    return run_full_codesign_pipeline(profile, top_n=20)

with t1:
    st.subheader("Human–AI Co-Designed Job Search & Decision Engine")
    
    with st.spinner("Executing 2-Stage Hybrid Search & 6-Factor Scoring Engine..."):
        ranked_df = get_pipeline_data(candidate_profile)
        
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        query = st.text_input("Filter by Title / Keywords", placeholder="e.g., Data Scientist, Healthcare, Analytics")
    with c2:
        loc_filter = st.selectbox("Filter Location", ["All"] + sorted(list(set(ranked_df["location"].dropna()))))
    with c3:
        remote_only = st.checkbox("Remote / Hybrid Only", True)
        
    filtered = ranked_df.copy()
    if query:
        filtered = filtered[filtered["title"].str.contains(query, case=False, na=False) | filtered["description"].str.contains(query, case=False, na=False)]
    if loc_filter != "All":
        filtered = filtered[filtered["location"] == loc_filter]
        
    if filtered.empty:
        st.warning("No jobs match your current search filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Candidates Filtered", f"{len(filtered)} / {len(ranked_df)}")
        m2.metric("Top Match %", f"{filtered.iloc[0]['match_percent']:.1f}%")
        m3.metric("Evaluation Coverage", f"{filtered.iloc[0]['evaluation_coverage_percent']:.1f}%")
        m4.metric("Hard Eligibility Flags", int(sum(filtered["eligibility_status"] == "Hard eligibility requirement present")))
        
        st.markdown("### Top Match Scores")
        st.bar_chart(filtered.head(8).set_index("title")["match_percent"])
        
        st.markdown("### Top Job Recommendations & Human Review Decision Override")
        st.info("Human-AI Co-Design Rule: AI Recommendation can be confirmed or overridden by the Human Reviewer below.")
        
        # Prepare Interactive Decision Editor
        display_cols = ["title", "company", "match_percent", "match_tier", "evaluation_coverage", "eligibility_status", "recommendation"]
        editor_df = filtered.head(10)[display_cols].copy()
        editor_df.rename(columns={
            "match_percent": "Match %",
            "match_tier": "Tier",
            "evaluation_coverage": "Coverage",
            "eligibility_status": "Eligibility Audit",
            "recommendation": "AI Recommendation"
        }, inplace=True)
        
        editor_df["Human Review Decision"] = editor_df["AI Recommendation"].apply(
            lambda r: st.session_state.human_decisions.get(r, "Accept AI")
        )
        
        edited_table = st.data_editor(
            editor_df,
            column_config={
                "Human Review Decision": st.column_config.SelectboxColumn(
                    "Human Review Decision",
                    help="Select human override decision",
                    options=["Accept AI", "Apply", "Consider", "Reject / Skip", "Needs Manual Review"],
                    required=True
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Compute Final Auditable Recommendation
        edited_table["Final Auditable Recommendation"] = edited_table.apply(
            lambda row: apply_human_decision(row["AI Recommendation"], row["Human Review Decision"]), axis=1
        )
        
        st.markdown("#### Final Auditable Decision Summary")
        st.dataframe(edited_table[["title", "Match %", "AI Recommendation", "Human Review Decision", "Final Auditable Recommendation"]], use_container_width=True, hide_index=True)
        
        st.markdown("### Detailed Explanations for Top Ranked Jobs")
        for idx, row in filtered.head(5).iterrows():
            with st.expander(f"{row['title']} — {row['company']} • Match: {row['match_percent']:.1f}% ({row['match_tier']})"):
                st.markdown(f"**Location / Work Mode**: {row['location']}")
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
                    st.write(f"- Skills: `{row['skills_score']*100:.1f}%`")
                    st.write(f"- Experience: `{row['experience_projects_score']*100:.1f}%`")
                    st.write(f"- Education: `{row['education_score']*100:.1f}%`")
                    st.write(f"- Role Alignment: `{row['role_alignment_score']*100:.1f}%`")
                    st.write(f"- Location/Work: `{row['location_work_score']*100:.1f}%`")
                    
                st.divider()
                st.markdown("**Job Description Snippet:**")
                st.caption(str(row["description"])[:500] + "...")

with t2:
    st.subheader("Stage 1: Hugging Face 2-Stage Hybrid Retrieval Engine")
    st.markdown("Combines BM25 lexical search with `sentence-transformers/all-MiniLM-L6-v2` dense embeddings ($50\\% \\text{ BM25} + 50\\% \\text{ Semantic}$).")
    
    if 'ranked_df' in locals() and not ranked_df.empty:
        hybrid_display = ranked_df[["title", "company", "bm25_score", "semantic_score", "hybrid_score"]].head(15).copy()
        hybrid_display["BM25 Raw"] = hybrid_display["bm25_score"].round(2)
        hybrid_display["Semantic Similarity"] = (hybrid_display["semantic_score"] * 100).round(1).astype(str) + "%"
        hybrid_display["Hybrid Score"] = (hybrid_display["hybrid_score"] * 100).round(1).astype(str) + "%"
        st.dataframe(hybrid_display[["title", "company", "BM25 Raw", "Semantic Similarity", "Hybrid Score"]], use_container_width=True, hide_index=True)

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

with t4:
    st.subheader("Manual vs. AI Experiment Matrix (CS 5588 Benchmark)")
    st.markdown("Tracks historical progression from manual human design baseline to intermediate AI design, ending at the validated co-designed system.")
    edited_comp = st.data_editor(st.session_state.comparison, use_container_width=True, hide_index=True, num_rows="fixed")
    st.session_state.comparison = edited_comp
    st.download_button("Download Comparison CSV", edited_comp.to_csv(index=False), "manual_vs_ai_comparison.csv", "text/csv")

with t5:
    st.subheader("Agentic AI + GitHub Traceability Workflow")
    st.markdown("**GOAL → PLAN → TOOLS → OBSERVE → REVISE → VERIFY**")
    prompt = '''We are building a teaching application for AI-guided job search.
Before changing code, propose a short plan.

Requirements:
1. Keep every score in [0,1].
2. Preserve separate evidence for skills, experience, education, role alignment, location, and salary.
3. Validate hard eligibility outside weighted numerical match score.
4. Do not invent candidate qualifications or job requirements.
5. Provide auditable human decision override controls.
'''
    st.text_area("Agentic AI System Prompt", prompt, height=220)
    st.markdown("#### GitHub Experiment Branches")
    st.code('''git checkout -b human-baseline
git commit -m "HUMAN: add transparent baseline matcher"

git checkout -b ai-guided-matcher
git commit -m "AI-GENERATED: propose improved matching logic"

git checkout -b co-designed
git commit -m "CO-DESIGNED: verify and correct matching logic"''', language="bash")

st.divider()
st.caption("CS 5588 Challenge 1 • Validated Teaching Prototype • Keep human review in the loop.")
