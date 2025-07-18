import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz.fuzz import token_sort_ratio
import re

st.set_page_config(layout="centered")

# === Styling ===
st.markdown("""
<style>
body, .stApp, .block-container {
    background-color: #000;
    color: #fff;
}
.stTextInput input, .stFileUploader input {
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

# === Header ===
st.markdown("<h2 style='text-align:center;color:#fff;'>Hiscox Digital Atlas: Vague Match Mode</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border-top:2px solid #d7263d;'>", unsafe_allow_html=True)

# === Load Backbone Map ===
@st.cache_data
def load_backbone():
    df = pd.read_excel("Atlas COB NAICS Map.xlsx", sheet_name="Complete COB Map")
    df["match_corpus"] = (
        df["Hiscox_COB"].astype(str).str.strip() + " | " +
        df["NAICS_Description"].astype(str).str.strip() + " | " +
        df["COB_Group"].astype(str).str.strip()
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

# === Input Cleaning ===
def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# === Scoring Function with Vague Mode Enhancements ===
def compute_score(input_text, match_row, similarity_score):
    score = 0.0
    sources = []

    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    naics_clean = normalize(match_row["NAICS_Description"])
    group_clean = normalize(match_row["COB_Group"])

    # === Consulting override ===
    if "consulting" in input_clean:
        if "consulting" in cob_clean or "consulting" in group_clean:
            score += 0.4
            sources.append("Consulting Override Boost")
        elif match_row["Hiscox_COB"] in ["Packaging Services", "Retail Sales", "Manufacturing", "Logistics"]:
            score -= 0.3
            sources.append("Consulting Mismatch Penalty")

    # === Exact match to COB
    if input_clean == cob_clean:
        score += 1.0
        sources.append("Exact COB Match")

    # === Fuzzy COB match
    fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])
    if fuzzy_score > 85:
        score += 0.8
        sources.append(f"Fuzzy COB Match ({fuzzy_score})")

    # === Keyword overlap: NAICS
    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0:
        score += 0.3
        sources.append("Keyword NAICS Match")

    # === Keyword overlap: COB Group
    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0:
        score += 0.2
        sources.append("COB Group Keyword Match")

    # === Semantic similarity
    score += similarity_score * 0.6
    sources.append("Semantic Similarity")

    return round(score, 4), "; ".join(sources)

# === Match Engine ===
def run_batch_match(inputs, threshold=0.4):
    results = []
    inputs = inputs.fillna("").astype(str).str.strip()
    progress = st.progress(0)

    for i, entry in enumerate(inputs):
        embedding = model.encode([entry], convert_to_numpy=True, show_progress_bar=False)
        D, I = index.search(embedding, 1)
        best_idx = I[0][0]
        dist = D[0][0]
        similarity = 1 / (1 + dist)

        match_row = engine_df.iloc[best_idx]
        score, sources = compute_score(entry, match_row, similarity)

        results.append({
            "Input_Description": entry,
            "Hiscox_COB": match_row["Hiscox_COB"],
            "NAICS_Code": match_row["NAICS_Code"],
            "NAICS_Description": match_row["NAICS_Description"],
            "COB_Group": match_row["COB_Group"],
            "PL": match_row["PL"],
            "GL": match_row["GL"],
            "BOP": match_row["BOP"],
            "Cyber": match_row["Cyber"],
            "Similarity_Score": round(similarity, 4),
            "Composite_Score": score,
            "Match_Sources": sources,
            "Confidence_Level": "Low" if score < threshold else "OK"
        })

        progress.progress((i + 1) / len(inputs))

    return pd.DataFrame(results)

# === Upload Interface ===
st.markdown("### 📥 Upload Partner Descriptions to Match")
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])
confidence_cutoff = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    if "Partner_Description" not in df.columns:
        st.error("Your file must contain a column named 'Partner_Description'.")
    else:
        with st.spinner("Running semantic and fuzzy matching..."):
            result_df = run_batch_match(df["Partner_Description"], threshold=confidence_cutoff)

        st.success("Matching complete.")
        st.dataframe(result_df.head(15), use_container_width=True)

        st.download_button(
            "⬇️ Download Match Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="vague_mode_scored_matches.csv",
            mime="text/csv"
        )
