
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# === Placeholder Search Bar ===
st.text_input("🔍 Search Business Description (coming soon)", placeholder="e.g., Landscaping, IT Consulting")

@st.cache_data
def load_backbone():
    df = pd.read_excel("Atlas COB NAICS Map.xlsx", sheet_name="Complete COB Map")
    df["NAICS_Description"] = df["NAICS_Description"].astype(str).str.strip()
    return df

@st.cache_resource
def load_model_and_index(descriptions):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions.tolist(), convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index, embeddings

engine_df = load_backbone()
model, index, engine_embeddings = load_model_and_index(engine_df["NAICS_Description"])

# === FAISS Batch Match Function ===
def batch_match_with_faiss(input_descriptions, batch_size=512):
    results = []
    input_descriptions = [str(d).strip() for d in input_descriptions]
    total = len(input_descriptions)
    progress = st.progress(0)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = input_descriptions[start:end]
        input_embeds = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        D, I = index.search(input_embeds, 1)

        for i, idx in enumerate(I):
            best_match_idx = idx[0]
            score = 1 / (1 + D[i][0])
            row = engine_df.iloc[best_match_idx]

            results.append({
                "NAICS_Code": row["NAICS_Code"],
                "Input_Description": batch[i],
                "Matched_Description": row["NAICS_Description"],
                "Similarity_Score": round(score, 4),
                "Hiscox_COB": row["Hiscox_COB"],
                "Mapping_ID": row.get("Mapping_ID", ""),
                "PL": row["PL"],
                "GL": row["GL"],
                "BOP": row["BOP"],
                "Cyber": row["Cyber"],
                "COB_Group": row["COB_Group"]
            })

        progress.progress(min(end / total, 1.0))

    return pd.DataFrame(results)

# === Upload Partner File ===
st.markdown("### 📥 Upload a Partner File")

with st.expander("📄 View file format instructions (click here)", expanded=False):
    st.markdown("The input file should include a column labeled `NAICS_Description`. This may change. For now, use [this placeholder guide](https://example.com/naics-input-format) for details.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "NAICS_Description" not in df.columns:
        st.error("Uploaded file must contain a 'NAICS_Description' column.")
    else:
        st.success("Running FAISS-based matching in optimized batches...")
        with st.spinner("Matching in progress... please wait"):
            result_df = batch_match_with_faiss(df["NAICS_Description"])

        # Reorder columns
        column_order = [
            "NAICS_Code", "Input_Description", "Matched_Description", "Similarity_Score",
            "Hiscox_COB", "Mapping_ID", "PL", "GL", "BOP", "Cyber", "COB_Group"
        ]
        result_df = result_df[column_order]

        st.dataframe(result_df.head(10), use_container_width=True)

        st.download_button(
            "⬇️ Download Matched Report",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="Digital_Atlas_Report_Ordered.csv",
            mime="text/csv"
        )
