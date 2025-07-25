import pandas as pd
import streamlit as st
import numpy as np
import faiss
import time
import threading
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="centered")

# === Custom CSS for true black background and red/white accents ===
# === Custom CSS + JS for Black Theme and Drag-Over Highlight ===
st.markdown("""
    <style>
        body, .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stTextInput > div > div > input {
            background-color: #222222;
            color: #ffffff;
        }
        .stFileUploader, .stButton {
            background-color: #111111;
        }
        .drop-target-active {
            border: 2px dashed #ff4b4b !important;
            background-color: #1a1a1a !important;
            border-radius: 8px !important;
        }
    </style>
    <script>
        const observer = new MutationObserver(() => {
            const dropzone = document.querySelector('section[data-testid="stFileUploader"]');
            if (dropzone && !dropzone.classList.contains("watched")) {
                dropzone.classList.add("watched");

                // Add drag events
                dropzone.addEventListener("dragenter", function () {
                    dropzone.classList.add("drop-target-active");
                });
                dropzone.addEventListener("dragleave", function () {
                    dropzone.classList.remove("drop-target-active");
                });
                dropzone.addEventListener("drop", function () {
                    dropzone.classList.remove("drop-target-active");
                });
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_excel("AtlasEngine.xlsx", sheet_name="NAICS_COB")
    for col in ["Hiscox_COB", "NAICS_Title", "NAICS_Description", "COB_Group"]:
        df[col] = df[col].astype(str).str.strip()
    df["match_corpus"] = (
        df["NAICS_Description"] + " | " +
        df["NAICS_Title"] + " | " +
        df["COB_Group"] + " | " +
        df["Hiscox_COB"]
    )
    return df

engine_df = load_data()

# === LOAD MODEL ===
@st.cache_resource
def build_model_index(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

model, index = build_model_index(engine_df["match_corpus"])

# === MATCHING HELPERS ===
def summarize_appetite_logic(row):
    flags = [row["PL"], row["GL"], row["BOP"], row["Cyber"]]
    labels = ["PL", "GL", "BOP", "Cyber"]
    yes_flags = [label for flag, label in zip(flags, labels) if flag == "Y"]
    if len(yes_flags) >= 2:
        return "In Appetite"
    elif len(yes_flags) == 1:
        return f"{yes_flags[0]} Only"
    else:
        return "Out of Appetite"

def confidence_label(score_pct):
    if score_pct >= 86:
        return f"{score_pct:.1f}% (High Confidence)"
    elif score_pct >= 70:
        return f"{score_pct:.1f}% (Needs Review)"
    else:
        return f"{score_pct:.1f}% (Low Confidence)"

def match_status(score_pct):
    if score_pct >= 86:
        return "Confirmed"
    elif score_pct >= 70:
        return "Needs Review"
    else:
        return "No Match Found"

def check_tier1_rules(input_clean, engine_df):
    cob_match = engine_df[engine_df["Hiscox_COB"].str.lower() == input_clean]
    if not cob_match.empty:
        return cob_match.iloc[0], "Tier 1 Match: Exact Hiscox COB"
    naics_match = engine_df[engine_df["NAICS_Description"].str.lower() == input_clean]
    if not naics_match.empty:
        return naics_match.iloc[0], "Tier 1 Match: Exact NAICS Description"
    return None, None

# === SEARCH INTERFACE ===
def search_top_matches(input_text):
    input_clean = str(input_text).strip().lower()

    # === NAICS SEARCH HANDLING ===
    if input_clean.isdigit() and len(input_clean) == 6:
        naics_matches = engine_df[engine_df["NAICS_Code"].astype(str).str.startswith(input_clean)]
        results = []
        for _, row in naics_matches.iterrows():
            results.append({
                "Input_Description": input_text,
                "Hiscox_COB": row["Hiscox_COB"],
                "full_industry_code": row.get("full_industry_code", ""),
                "Confidence (%)": "N/A (NAICS Search)",
                "Match_Status": "Confirmed via NAICS",
                "Appetite": summarize_appetite_logic(row),
                "PL": row["PL"],
                "GL": row["GL"],
                "BOP": row["BOP"],
                "Cyber": row["Cyber"]
            })
        return results[:3]

    # === TIER 1 RULES ===
    tier1_row, _ = check_tier1_rules(input_clean, engine_df)
    if tier1_row is not None:
        return [{
            "Input_Description": input_text,
            "Hiscox_COB": tier1_row["Hiscox_COB"],
            "full_industry_code": tier1_row.get("full_industry_code", ""),
            "Confidence (%)": "100.0% (High Confidence)",
            "Match_Status": "Confirmed",
            "Appetite": summarize_appetite_logic(tier1_row),
            "PL": tier1_row["PL"],
            "GL": tier1_row["GL"],
            "BOP": tier1_row["BOP"],
            "Cyber": tier1_row["Cyber"]
        }]

    # === SEMANTIC SEARCH ===
    embedding = model.encode([input_text], convert_to_numpy=True)
    D, I = index.search(embedding, 3)
    results = []
    for rank in range(3):
        idx = I[0][rank]
        row = engine_df.iloc[idx]
        score = 1 / (1 + D[0][rank])
        score_pct = round(score * 100, 1)
        results.append({
            "Input_Description": input_text,
            "Hiscox_COB": row["Hiscox_COB"],
            "full_industry_code": row.get("full_industry_code", ""),
            "Confidence (%)": confidence_label(score_pct),
            "Match_Status": match_status(score_pct),
            "Appetite": summarize_appetite_logic(row),
            "PL": row["PL"],
            "GL": row["GL"],
            "BOP": row["BOP"],
            "Cyber": row["Cyber"]
        })
    return results

# === BATCH MATCH ===
def run_batch_match(inputs):
    results = []
    embeddings = model.encode(inputs, convert_to_numpy=True)
    D, I = index.search(embeddings, 1)
    for i, entry in enumerate(inputs):
        input_clean = str(entry).strip().lower()
        tier1_row, _ = check_tier1_rules(input_clean, engine_df)
        if tier1_row is not None:
            results.append({
                "Input_Description": entry,
                "Hiscox_COB": tier1_row["Hiscox_COB"],
                "full_industry_code": tier1_row.get("full_industry_code", ""),
                "Confidence (%)": "100.0% (High Confidence)",
                "Match_Status": "Confirmed",
                "Appetite": summarize_appetite_logic(tier1_row),
                "PL": tier1_row["PL"],
                "GL": tier1_row["GL"],
                "BOP": tier1_row["BOP"],
                "Cyber": tier1_row["Cyber"]
            })
            continue

        idx = I[i][0]
        row = engine_df.iloc[idx]
        sim_score = 1 / (1 + D[i][0])
        sim_pct = round(sim_score * 100, 1)
        results.append({
            "Input_Description": entry,
            "Hiscox_COB": row["Hiscox_COB"],
            "full_industry_code": row.get("full_industry_code", ""),
            "Confidence (%)": confidence_label(sim_pct),
            "Match_Status": match_status(sim_pct),
            "Appetite": summarize_appetite_logic(row),
            "PL": row["PL"],
            "GL": row["GL"],
            "BOP": row["BOP"],
            "Cyber": row["Cyber"]
        })
    return pd.DataFrame(results)

# === UI ===
from PIL import Image

# === Display Logo at Top Center ===
logo = Image.open("AtlasLogo.jpeg")
st.image(logo, use_container_width=False, width=300)

# Top 3 Search
search_input = st.text_input("🔍 Search for a business description")
if search_input:
    st.markdown("#### Top 3 Matches:")
    for res in search_top_matches(search_input):
        st.markdown(f"- **{res['Hiscox_COB']}** — {res['Confidence (%)']} — *{res['Match_Status']}*")

st.markdown("---")
st.markdown("###  Batch Class of Business Mapping")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    text_column = None
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 5:
            text_column = col
            break

    if text_column:
        status_placeholder = st.empty()
        status_placeholder.markdown("### ⏳ Matching in progress...")
        result_df = run_batch_match(df[text_column].fillna(""))
        status_placeholder.success("✅ Matching complete.")

        st.dataframe(result_df.head(25), use_container_width=True)
        st.download_button(
            "⬇️ Download Match Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="Atlas_Match_Results.csv",
            mime="text/csv"
        )
    else:
        st.error("No suitable text column found in uploaded file.")
