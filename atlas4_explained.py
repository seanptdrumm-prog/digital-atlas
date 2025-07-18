
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
    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    naics_clean = normalize(match_row["NAICS_Description"])
    group_clean = normalize(match_row["COB_Group"])

    text_match = 0.0
    semantic = round(similarity_score * 0.6, 4)
    keyword = 0.0
    override = 0.0

    if input_clean == cob_clean:
        text_match = 1.0
    else:
        fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])
        if fuzzy_score > 85:
            text_match = 0.8

    if "consulting" in input_clean:
        if not ("consulting" in cob_clean or "consulting" in group_clean):
            override -= 0.3
        else:
            override += 0.4

    if "appraisal" in input_clean or "appraiser" in input_clean:
        override -= 0.5

    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0:
        keyword += 0.2

    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0:
        keyword += 0.1

    final_score = round(text_match + semantic + keyword + override, 4)
    return final_score, {
        "Text Match": text_match,
        "Semantic": semantic,
        "Keyword": keyword,
        "Overrides": override,
        "Final": final_score
    }

def get_confidence_bucket(score):
    if score >= 0.9:
        return "In Appetite", "✅", "limegreen", "High Confidence"
    elif 0.71 <= score < 0.9:
        return "Needs Review", "⚠️", "#FFCC00", "Medium Confidence"
    else:
        return "Out of Appetite", "❌", "#d7263d", "Low Confidence"

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
        score, _ = compute_score(entry, match_row, similarity)

        results.append({
            "Input_Description": entry,
            "Hiscox_COB": match_row["Hiscox_COB"],
            "Confidence_Level": "Low" if score < threshold else "OK",
            "Map_Code": match_row.get("Mapping_ID", ""),
            "COB_Group": match_row["COB_Group"],
            "PL": match_row["PL"],
            "GL": match_row["GL"],
            "BOP": match_row["BOP"],
            "Cyber": match_row["Cyber"]
        })

        progress.progress((i + 1) / len(inputs))

    return pd.DataFrame(results)

# === SEARCH UI ===
st.markdown("<h2 style='text-align:center;color:#fff;'>Atlas: Search COBs by Description</h2>", unsafe_allow_html=True)
user_input = st.text_input("🔍 Enter a business description", placeholder="e.g., Appraisal services, IT consulting, Food truck")

if user_input.strip():
    input_embed = model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(input_embed, 1)
    best_idx = I[0][0]
    best_row = engine_df.iloc[best_idx]
    dist = D[0][0]
    similarity = 1 / (1 + dist)
    score, breakdown = compute_score(user_input, best_row, similarity)
    conf_bucket, icon, color, label = get_confidence_bucket(score)

    lobs = {
        "PL": str(best_row["PL"]).strip().upper(),
        "GL": str(best_row["GL"]).strip().upper(),
        "BOP": str(best_row["BOP"]).strip().upper(),
        "Cyber": str(best_row["Cyber"]).strip().upper()
    }
    lob_tooltip = "\n".join([f"{lob}: {'✅' if val == 'Y' else '❌'} ({val})" for lob, val in lobs.items()])

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #111 0%, #000 100%);
        border-left: 6px solid {color};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        color: #fff;'
        title="{lob_tooltip}">
      <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div style='font-size: 22px; font-weight: bold;'>{best_row["Hiscox_COB"]}</div>
        <button style='background-color:{color};color:#000;border:none;padding:6px 12px;border-radius:6px;font-weight:bold;cursor:pointer;' title="{lob_tooltip}">{conf_bucket}</button>
      </div>
      <div style='font-size: 16px; color: #ccc;'>NAICS: {best_row["NAICS_Description"]}</div>
      <div style='margin-top: 10px;'>
        <b>NAICS Code:</b> {best_row["NAICS_Code"]}<br>
        <b>COB Group:</b> {best_row["COB_Group"]}
      </div>
      <div style='margin-top: 12px; font-size: 18px; font-weight: bold;'>
        <span style='color:{color}'>Composite Score: {score} / 2.5 → {label}</span>
        <ul style='font-size:14px; color:#ccc; margin-top:10px;'>
          <li>Text Match: {breakdown["Text Match"]}</li>
          <li>Semantic Similarity: {breakdown["Semantic"]}</li>
          <li>Keyword Match: {breakdown["Keyword"]}</li>
          <li>Overrides: {breakdown["Overrides"]}</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

# === BATCH MODE ===
st.markdown("### 📥 Or Upload a File")
uploaded_file = st.file_uploader("Upload CSV or Excel with 'Partner_Description' column", type=["csv", "xlsx"])
confidence_cutoff = 0.4

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Partner_Description" not in df.columns:
        st.error("File must contain a 'Partner_Description' column.")
    else:
        with st.spinner("Scoring and matching... please wait"):
            result_df = run_batch_match(df["Partner_Description"], threshold=confidence_cutoff)

        st.success("Results ready:")
        st.dataframe(result_df.head(15), use_container_width=True)
        st.download_button(
            "⬇️ Download Full Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="atlas4_results.csv",
            mime="text/csv"
        )
