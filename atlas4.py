import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz.fuzz import token_sort_ratio

st.set_page_config(layout="centered")

# === STYLING ===
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

# === LOAD DATA ===
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

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def compute_score(input_text, match_row, similarity_score):
    score = 0.0
    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    group_clean = normalize(match_row["COB_Group"])

    if "consulting" in input_clean:
        if not ("consulting" in cob_clean or "consulting" in group_clean):
            score -= 0.3

    if input_clean == cob_clean:
        score += 1.0

    fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])
    if fuzzy_score > 85:
        score += 0.8

    naics_clean = normalize(match_row["NAICS_Description"])
    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0:
        score += 0.3

    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0:
        score += 0.2

    score += similarity_score * 0.6
    return round(score, 4)

def get_confidence_bucket(score):
    if score >= 0.9:
        return "In Appetite", "✅", "limegreen", "High Confidence"
    elif 0.71 <= score < 0.9:
        return "Needs Review", "⚠️", "#FFCC00", "Medium Confidence"
    else:
        return "Out of Appetite", "❌", "#d7263d", "Low Confidence"

# === SEARCH BAR ===
st.markdown("<h2 style='text-align:center;color:#fff;'>Atlas: Search COBs by Description</h2>", unsafe_allow_html=True)
user_input = st.text_input("🔍 Enter a business description", placeholder="e.g., Appraisal services, IT consulting, Food truck")

if user_input.strip():
    input_embed = model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(input_embed, 1)
    best_idx = I[0][0]
    best_row = engine_df.iloc[best_idx]
    dist = D[0][0]
    similarity = 1 / (1 + dist)
    score = compute_score(user_input, best_row, similarity)
    conf_bucket, icon, color, label = get_confidence_bucket(score)

    lobs = {
        "PL": best_row["PL"] == "Y",
        "GL": best_row["GL"] == "Y",
        "BOP": best_row["BOP"] == "Y",
        "Cyber": best_row["Cyber"] == "Y"
    }
    lob_status = "\n".join([f"{lob}: {'✅' if ok else '❌'}" for lob, ok in lobs.items()])
    match_name = best_row["Hiscox_COB"]

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #111 0%, #000 100%);
        border-left: 6px solid {color};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        color: #fff;'
        title="{lob_status}">
      <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div style='font-size: 22px; font-weight: bold;'>{match_name}</div>
        <button style='background-color:{color};color:#000;border:none;padding:6px 12px;border-radius:6px;font-weight:bold;cursor:pointer;' title="{lob_status}">{conf_bucket}</button>
      </div>
      <div style='font-size: 16px; color: #ccc;'>NAICS: {best_row["NAICS_Description"]}</div>
      <div style='margin-top: 10px;'>
        <b>NAICS Code:</b> {best_row["NAICS_Code"]}<br>
        <b>COB Group:</b> {best_row["COB_Group"]}
      </div>
      <div style='margin-top: 12px; font-size: 18px; font-weight: bold;'>
        <span style='color:{color}'>{int(score * 100)}% Match Confidence ({label})</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# === BATCH MODE (still available) ===
st.markdown("### 📥 Or Upload a File")
uploaded_file = st.file_uploader("Upload CSV or Excel with 'Partner_Description' column", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Partner_Description" not in df.columns:
        st.error("File must contain a 'Partner_Description' column.")
    else:
        st.success("File uploaded! Preview below:")
        st.dataframe(df.head(5), use_container_width=True)
