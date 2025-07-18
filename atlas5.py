
import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz.fuzz import token_sort_ratio

st.set_page_config(layout="centered")

@st.cache_data
def load_data():
    complete_df = pd.read_excel("Atlas COB NAICS Map.xlsx", sheet_name="Complete COB Map")
    api_df = pd.read_excel("Atlas COB NAICS Map.xlsx", sheet_name="API")
    complete_df["match_corpus"] = (
        complete_df["Hiscox_COB"].astype(str).str.strip() + " | " +
        complete_df["NAICS_Description"].astype(str).str.strip() + " | " +
        complete_df["COB_Group"].astype(str).str.strip()
    )
    return complete_df, api_df

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

def normalize(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower().strip())

def compute_score(input_text, match_row, similarity_score):
    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    naics_clean = normalize(match_row["NAICS_Description"])
    group_clean = normalize(match_row["COB_Group"])

    score = 0.0
    fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])

    if input_clean == cob_clean:
        score += 1.0
    elif fuzzy_score > 85:
        score += 0.8

    if "consulting" in input_clean:
        if "consulting" in cob_clean or "consulting" in group_clean:
            score += 0.4
        else:
            score -= 0.3

    if "appraisal" in input_clean or "appraiser" in input_clean:
        score -= 0.5

    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0:
        score += 0.2

    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0:
        score += 0.1

    score += similarity_score * 0.6
    return round(score, 4)

def fallback_match(input_text):
    input_norm = normalize(input_text)
    for i, row in api_df.iterrows():
        api_cob = str(row["Hiscox_COB"]).strip()
        if input_norm in normalize(api_cob):
            return row
    return None

def run_match_batch(inputs, threshold=0.4):
    results = []
    inputs = inputs.fillna("").astype(str).str.strip()
    progress = st.progress(0)

    for i, entry in enumerate(inputs):
        embedding = model.encode([entry], convert_to_numpy=True, show_progress_bar=False)
        D, I = index.search(embedding, 1)
        best_idx = I[0][0]
        dist = D[0][0]
        similarity = 1 / (1 + dist)
        match_row = complete_df.iloc[best_idx]
        score = compute_score(entry, match_row, similarity)

        if score < threshold:
            fallback_row = fallback_match(entry)
            if fallback_row is not None:
                match_row = fallback_row

        results.append({
            "Input_Description": entry,
            "Hiscox_COB": match_row["Hiscox_COB"],
            "COB_Group": match_row.get("COB_Group", ""),
            "PL": match_row["PL"],
            "GL": match_row["GL"],
            "BOP": match_row["BOP"],
            "Cyber": match_row["Cyber"],
            "Confidence_Level": "Low" if score < threshold else "OK"
        })
        progress.progress((i + 1) / len(inputs))

    return pd.DataFrame(results)

# === UI ===
st.markdown("<h2 style='text-align:center;color:#fff;'>Atlas 5: Enhanced Matching Engine</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border-top:2px solid #d7263d;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a Partner Description File", type=["csv", "xlsx"])
confidence_cutoff = 0.4

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Partner_Description" not in df.columns:
        st.error("File must contain a 'Partner_Description' column.")
    else:
        with st.spinner("Matching and scoring..."):
            result_df = run_match_batch(df["Partner_Description"], threshold=confidence_cutoff)

        st.success("Done!")
        st.dataframe(result_df.head(10), use_container_width=True)

        st.download_button(
            "⬇️ Download Matched Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="atlas5_results.csv",
            mime="text/csv"
        )
