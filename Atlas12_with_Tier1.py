import pandas as pd
import streamlit as st
import numpy as np
import faiss
import time
import threading
from sentence_transformers import SentenceTransformer

# === PAGE CONFIG ===
st.set_page_config(layout="centered")

# === SIMS LOADING ANIMATION ===
sims_messages = [
    "Reticulating splines...",
    "Calculating confidence coefficient...",
    "Greasing the underwriting gears...",
    "Matching vibes with business types...",
    "Aligning chakras and coverage terms...",
    "Fuzzifying reality...",
    "Summoning digital underwriter spirit...",
    "Tuning semantic antenna...",
    "Consulting the NAICS oracle...",
    "Tetris-ing your input descriptions..."
]

def loading_animation(stop_signal, placeholder):
    i = 0
    while not stop_signal["stop"]:
        msg = sims_messages[i % len(sims_messages)]
        placeholder.markdown(f"### ⏳ {msg}")
        time.sleep(2.5)
        i += 1

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_excel("AtlasEngine.xlsx", sheet_name="NAICS_COB")
    for col in ["Hiscox_COB", "NAICS_Title", "NAICS_Description", "COB_Group"]:
        df[col] = df[col].astype(str).str.strip()
    df["match_corpus"] = (
        df["Hiscox_COB"] + " | " +
        df["NAICS_Title"] + " | " +
        df["NAICS_Description"] + " | " +
        df["COB_Group"]
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

# === TIER 1 RULES ===
def check_tier1_rules(input_clean, engine_df):
    cob_match = engine_df[engine_df["Hiscox_COB"].str.lower() == input_clean]
    if not cob_match.empty:
        return cob_match.iloc[0], "Tier 1 Match: Exact Hiscox COB"
    naics_match = engine_df[engine_df["NAICS_Description"].str.lower() == input_clean]
    if not naics_match.empty:
        return naics_match.iloc[0], "Tier 1 Match: Exact NAICS Description"
    return None, None

# === APPETITE SUMMARY ===
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

# === CONFIDENCE INTERPRETER ===
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

# === RUN BATCH MATCH ===
def run_batch_match(inputs):
    results = []
    embeddings = model.encode(inputs, convert_to_numpy=True)
    D, I = index.search(embeddings, 1)

    for i, entry in enumerate(inputs):
        input_clean = str(entry).strip().lower()
        tier1_row, tier1_reason = check_tier1_rules(input_clean, engine_df)

        if tier1_row is not None:
            result = {
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
            }
            results.append(result)
            continue

        idx = I[i][0]
        match_row = engine_df.iloc[idx]
        sim_score = 1 / (1 + D[i][0])
        sim_pct = round(sim_score * 100, 1)

        result = {
            "Input_Description": entry,
            "Hiscox_COB": match_row["Hiscox_COB"],
            "full_industry_code": match_row.get("full_industry_code", ""),
            "Confidence (%)": confidence_label(sim_pct),
            "Match_Status": match_status(sim_pct),
            "Appetite": summarize_appetite_logic(match_row),
            "PL": match_row["PL"],
            "GL": match_row["GL"],
            "BOP": match_row["BOP"],
            "Cyber": match_row["Cyber"]
        }
        results.append(result)

    return pd.DataFrame(results)

# === UI ===
st.image("AtlasLogo.jpeg", use_column_width=True)
st.markdown("### 📥 Upload Partner Descriptions")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    text_column = None
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 5:
            text_column = col
            break

    if text_column:
        st.success(f"Using column: '{text_column}'")
        status_placeholder = st.empty()
        stop_signal = {"stop": False}
        t = threading.Thread(target=loading_animation, args=(stop_signal, status_placeholder))
        t.start()

        try:
            result_df = run_batch_match(df[text_column].fillna(""))
        finally:
            stop_signal["stop"] = True
            t.join()

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