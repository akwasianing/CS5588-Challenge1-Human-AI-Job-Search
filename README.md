# CS 5588 Challenge 1 — Streamlit Job Search Application

This companion application turns the Challenge 1 Jupyter/Colab notebook into an interactive Streamlit prototype.

## Features

- Editable user profile
- Job search and filters
- Explainable weighted ranking
- Skill-gap explanations
- Optional Hugging Face semantic matching
- Hybrid baseline + semantic ranking
- Feedback logging
- Manual vs. AI-guided experiment table
- Google Antigravity agentic-development prompt
- GitHub traceability workflow

The starter uses **synthetic job postings**. Replace them only with instructor-approved data or APIs.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit normally opens the app at `http://localhost:8501`.

## Recommended GitHub layout

```text
job-search-challenge/
├── README.md
├── streamlit_app.py
├── requirements.txt
├── notebooks/
│   └── CS5588_Challenge1_AI_Guided_Job_Search_Starter_Streamlit.ipynb
├── data/
├── tests/
└── results/
```

Suggested experiment branches:

```bash
git checkout -b human-baseline
git checkout -b ai-guided-matcher
```

Suggested traceability labels: `HUMAN`, `AI-GENERATED`, `CO-DESIGNED`.

## Streamlit Community Cloud

Push `streamlit_app.py` and `requirements.txt` to GitHub, select the repository in Streamlit Community Cloud, set the main file to `streamlit_app.py`, deploy, and verify the application before recording the URL in the project README.

## Hugging Face

The optional semantic matcher uses `sentence-transformers/all-MiniLM-L6-v2`. The first run may download model weights.

## Antigravity activity

Use the prompt in the **Agentic AI + GitHub** tab to ask the agent to improve a bounded component. Compare the human baseline, AI-generated change, and human-AI co-designed final version.

The goal is not to prove that AI always wins. The goal is to determine where agentic AI helps, where it fails, and what must remain under human review.
