
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fuzzywuzzy import fuzz

st.set_page_config(layout="centered")

# === STYLING ===
st.markdown("""
<style>
body, .stApp, .block-container {
    background-color: #000;
    color: #fff;
}
.stTextInput input {
    background: #222;
    color: #fff;
}
.stButton>button, .stDownloadButton>button {
    background-color: #d7263d;
    color: #fff;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #a0001e;
}
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h2 style='text-align:center;color:#fff;'>Hiscox Digital Atlas</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border-top:2px solid #d7263d;'>", unsafe_allow_html=True)

@st.cache_data
def load_backbone():
    df = pd.read_excel("Atlas COB NAICS Map.xlsx", sheet_name="Complete COB Map")
    df["NAICS_Description"] = df["NAICS_Description"].astype(str).str.strip()
    df["Hiscox_COB"] = df["Hiscox_COB"].astype(str).str.strip()
    df["COB_Group"] = df["COB_Group"].astype(str).str.strip()
    df["match_corpus"] = (
        df["Hiscox_COB"] + " | " +
        df["NAICS_Description"] + " | " +
        df["COB_Group"]
    )
    return df

@st.cache_resource
def load_model_and_index(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

engine_df = load_backbone()
model, index = load_model_and_index(engine_df["match_corpus"])

# === Composite Scoring Function ===
def compute_score(partner_desc, matched_row, similarity_score):
    score = 0.0
    sources = []

    # Exact match to Hiscox COB
    if partner_desc.strip().lower() == matched_row["Hiscox_COB"].strip().lower():
        score += 1.0
        sources.append("Exact COB Match")

    # Fuzzy match to Hiscox COB
    fuzzy_cob_score = fuzz.token_sort_ratio(partner_desc, matched_row["Hiscox_COB"])
    if fuzzy_cob_score > 85:
        score += 0.8
        sources.append("Fuzzy COB Match")

    # Match to NAICS Description
    if partner_desc.lower() in matched_row["NAICS_Description"].lower():
        score += 0.5
        sources.append("NAICS Desc Match")

    # Match to COB Group
    if partner_desc.lower() in matched_row["COB_Group"].lower():
        score += 0.2
        sources.append("COB Group Match")

    # Add vector similarity scaled
    score += similarity_score * 0.5
    sources.append("Semantic Similarity")

    return round(score, 4), "; ".join(sources)

# === Matching in Batches with Scoring ===
def batch_match_with_faiss(input_descriptions, batch_size=512):
    results = []
    input_descriptions = [str(d).strip() for d in input_descriptions]
    total = len(input_descriptions)
    progress = st.progress(0)

    for batch_num, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch = input_descriptions[start:end]
        input_embeds = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        D, I = index.search(input_embeds, 1)

        for i, idx in enumerate(I):
            best_match_idx = idx[0]
            semantic_score = 1 / (1 + D[i][0])
            row = engine_df.iloc[best_match_idx]
            total_score, source_hits = compute_score(batch[i], row, semantic_score)

            results.append({
                "Input_Description": batch[i],
                "Matched_Description": row["match_corpus"],
                "NAICS_Code": row["NAICS_Code"],
                "Hiscox_COB": row["Hiscox_COB"],
                "Mapping_ID": row.get("Mapping_ID", ""),
                "PL": row["PL"],
                "GL": row["GL"],
                "BOP": row["BOP"],
                "Cyber": row["Cyber"],
                "COB_Group": row["COB_Group"],
                "Similarity_Score": round(semantic_score, 4),
                "Composite_Score": total_score,
                "Match_Sources": source_hits
            })

        progress.progress(min((batch_num + 1) * batch_size / total, 1.0))

    return pd.DataFrame(results)

# === Upload Partner File ===
st.text_input("🔍 Search for a business description (coming soon)...", disabled=True)
st.markdown("### 📥 Upload a Partner Description File")
st.markdown("[What format do we need? (click to learn more)](#)", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Partner_Description" not in df.columns:
        st.error("Uploaded file must contain a 'Partner_Description' column.")
    else:
        st.success("Matching partner descriptions to Hiscox map...")
        with st.spinner("Scoring and matching... please wait"):
            result_df = batch_match_with_faiss(df["Partner_Description"])

        st.dataframe(result_df.head(10), use_container_width=True)

        st.download_button(
            "⬇️ Download Scored Matches",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="Digital_Atlas_Scored_Matches.csv",
            mime="text/csv"
        )
