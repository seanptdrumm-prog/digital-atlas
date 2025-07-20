# === Atlas 11: Hiscox Digital Atlas with New Engine, Logo, and Display Logic ===

import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz.fuzz import token_sort_ratio

st.set_page_config(layout="centered")

# === SECTION: Logo ===
st.markdown(
    "<div style='text-align:center;'><img src='AtlasLogo.jpeg' width='250'></div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border-top:2px solid #d7263d;'>", unsafe_allow_html=True)

# === SECTION: Load Data ===
@st.cache_data
def load_data():
    df1 = pd.read_excel("AtlasEngine.xlsx", sheet_name="Complete COB Map")
    df2 = pd.read_excel("AtlasEngine.xlsx", sheet_name="API")
    df1["match_corpus"] = df1["Hiscox_COB"].astype(str).str.strip() + " | " + df1["NAICS_Description"].astype(str).str.strip() + " | " + df1["COB_Group"].astype(str).str.strip()
    return df1, df2

@st.cache_resource
def load_model_and_index(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

complete_df, api_df = load_data()
model, index = load_model_and_index(complete_df["match_corpus"])

# === SECTION: Scoring + Matching Functions ===
def normalize(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower().strip())

def compute_score(input_text, match_row, similarity_score):
    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    naics_clean = normalize(match_row["NAICS_Description"])
    group_clean = normalize(match_row["COB_Group"])
    score = 0.0
    fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])
    if input_clean == cob_clean: score += 1.0
    elif fuzzy_score > 85: score += 0.8
    if "consulting" in input_clean:
        if "consulting" in cob_clean or "consulting" in group_clean: score += 0.4
        else: score -= 0.3
    if "appraisal" in input_clean or "appraiser" in input_clean: score -= 0.5
    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0: score += 0.2
    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0: score += 0.1
    score += similarity_score * 0.6
    return round(score, 4)

def fallback_match(input_text):
    input_norm = normalize(input_text)
    for _, row in api_df.iterrows():
        if input_norm in normalize(str(row["Hiscox_COB"])):
            return row
    return None

def get_confidence_band(score):
    if score >= 0.90: return "High"
    elif score >= 0.70: return "Needs Review"
    return "Low"

def summarize_appetite(row):
    lobs = {"PL": row["PL"], "GL": row["GL"], "BOP": row["BOP"], "Cyber": row["Cyber"]}
    yes_count = sum(1 for v in lobs.values() if v == "Y")
    nnb_flag = any(str(v).strip().upper() == "NNB" for v in lobs.values())
    cob_title = str(row.get("Hiscox_COB", "")).strip().lower()
    if "no appetite" in cob_title or cob_title in ["ooa", "out of appetite"]: return "Out of Appetite"
    if yes_count >= 2: return "In Appetite"
    elif yes_count == 1:
        for lob, v in lobs.items():
            if v == "Y": return f"{lob} Only"
    elif nnb_flag: return "Out of Appetite"
    return "Out of Appetite"

def transform_group_name(raw_group):
    if "OOA Industry" in raw_group:
        return "Out of Appetite: Industry"
    return raw_group

# === SECTION: Single Search ===
st.markdown("#### 🔎 Try a single description:")
user_input = st.text_input("Search COB", placeholder="e.g. Yoga, IT Consulting, Lawn Mowing")
if user_input:
    embedding = model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(embedding, 1)
    best_idx = I[0][0]
    dist = D[0][0]
    similarity = 1 / (1 + dist)
    match_row = complete_df.iloc[best_idx]
    score = compute_score(user_input, match_row, similarity)
    if score < 0.4:
        fallback_row = fallback_match(user_input)
        if fallback_row is not None:
            match_row = fallback_row
            score = 0.85
    confidence = get_confidence_band(score)
    appetite = summarize_appetite(match_row)
    display_group = transform_group_name(match_row['COB_Group'])
    color = {"High": "limegreen", "Needs Review": "#FFCC00", "Low": "#d7263d"}[confidence]
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #111 0%, #000 100%);
                    border-left: 6px solid {color}; border-radius: 12px;
                    padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.4); color: #fff;'>
          <div style='font-size: 22px; font-weight: bold;'>{match_row['Hiscox_COB']}</div>
          <div style='font-size: 16px; color: #ccc;'>Group: {display_group}</div>
          <div style='margin-top: 10px;'>
            <b>NAICS:</b> {match_row['NAICS_Code']}<br>
            <b>Description:</b> {match_row['NAICS_Description']}
          </div>
          <div style='margin-top: 12px; font-size: 16px;'>
            <span style='color:{color};font-weight:bold;'>{confidence} Confidence</span> &nbsp;|&nbsp;
            <span style='color:white;'>Appetite: <b>{appetite}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )