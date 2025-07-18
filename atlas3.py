import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz.fuzz import token_sort_ratio

st.set_page_config(layout="centered")

# Use Streamlit layout columns to center the logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("AtlasLogo.jpeg", width=250)

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
    sources = []
    input_clean = normalize(input_text)
    cob_clean = normalize(match_row["Hiscox_COB"])
    group_clean = normalize(match_row["COB_Group"])

    # Inspector disambiguation
    if "inspector" in input_clean or "inspection" in input_clean:
        if any(term in input_clean for term in ["home", "real estate", "property", "buyer", "seller"]):
            score -= 0.5
            sources.append("Penalty: Home/Real Estate Inspector")
        elif "insurance" in input_clean:
            score += 0.3
            sources.append("Boost: Insurance Inspector")
        elif any(term in input_clean for term in ["building", "compliance", "code"]):
            score += 0.2
            sources.append("Boost: Building Inspector")
        else:
            sources.append("Neutral: Generic Inspector (manual review advised)")

    # Auto work exclusion
    auto_terms = ["auto", "automobile", "vehicle", "car", "motor vehicle", "truck"]
    if any(term in input_clean for term in auto_terms):
        if any(tag in cob_clean for tag in ["no auto", "(no auto", "excluding auto", "non-auto"]):
            score -= 0.6
            sources.append("Penalty: Auto work matched to COB that excludes auto")
        else:
            sources.append("Auto-related description")

    # Truck vs food truck misclassification
    if "truck" in input_clean and "food" not in input_clean and "ice cream" not in input_clean:
        if group_clean == "mobile food services":
            score -= 0.6
            sources.append("Penalty: Commercial truck matched to food truck COB")

    # Boost for food stands and carts
    food_terms = ["hot dog", "popcorn", "beverage", "snack", "food stand", "food cart", "lunch wagon", "concession"]
    if any(term in input_clean for term in food_terms):
        if group_clean == "mobile food services":
            score += 0.4
            sources.append("Boost: Mobile Food Stand/Cart Language")

    # Home inspection NAICS boost
    home_inspection_terms = ["home inspection", "property inspection", "residential inspection", "real estate inspection", "house inspection"]
    if any(term in input_clean for term in home_inspection_terms):
        if "real estate" in cob_clean or "appraiser" in cob_clean:
            score += 0.4
            sources.append("Home Inspection Term → Boost to Real Estate Appraiser COB")

    # Expanded consulting logic
    if "consulting" in input_clean:
        if not ("consulting" in cob_clean or "consulting" in group_clean):
            score -= 0.3
            sources.append("Penalty: Consulting Misalignment")

    # Exact match
    if input_clean == cob_clean:
        score += 1.0
        sources.append("Exact COB Match")

    # Fuzzy match
    fuzzy_score = token_sort_ratio(input_text, match_row["Hiscox_COB"])
    if fuzzy_score > 85:
        score += 0.8
        sources.append(f"Fuzzy COB Match ({fuzzy_score})")

    # NAICS keyword overlap
    naics_clean = normalize(match_row["NAICS_Description"])
    keyword_overlap = len(set(input_clean.split()) & set(naics_clean.split()))
    if keyword_overlap > 0:
        score += 0.3
        sources.append("Keyword NAICS Match")

    # COB Group keyword overlap
    group_overlap = len(set(input_clean.split()) & set(group_clean.split()))
    if group_overlap > 0:
        score += 0.2
        sources.append("COB Group Keyword Match")

    # Semantic similarity
    score += similarity_score * 0.6
    sources.append("Semantic Similarity")

    return round(score, 4), "; ".join(sources)

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

st.markdown("### For Partners")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
# Confidence threshold is set internally.
# Adjust this value if you want to tune what counts as a "high confidence" match.
confidence_cutoff = 0.4


if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Partner_Description" not in df.columns:
        st.error("File must contain a 'Partner_Description' column.")
    else:
        with st.spinner("Matching and scoring..."):
            result_df = run_batch_match(df["Partner_Description"], threshold=confidence_cutoff)

        st.success("Done!")
        st.dataframe(result_df.head(15), use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="atlas_scored_results.csv",
            mime="text/csv"
        )
