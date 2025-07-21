
import pandas as pd
import streamlit as st
import numpy as np
import faiss
import time
import threading
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="centered")
st.title("🧭 Cartographer: Hiscox Classification Diagnostic Tool")

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

@st.cache_resource
def build_model_index(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

model, index = build_model_index(engine_df["match_corpus"])

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
        return "High Confidence"
    elif score_pct >= 70:
        return "Needs Review"
    else:
        return "Low Confidence"

def match_status(score_pct):
    if score_pct >= 86:
        return "Confirmed"
    elif score_pct >= 70:
        return "Needs Review"
    else:
        return "No Match Found"

def explain_confidence(score, input_text):
    if len(input_text) < 5:
        return "Input is too short"
    if len(input_text) > 100:
        return "Input is unusually long"
    if score < 0.5:
        return "Low semantic similarity"
    return "Standard match behavior"

def check_tier1_rules(input_clean, engine_df):
    cob_match = engine_df[engine_df["Hiscox_COB"].str.lower() == input_clean]
    if not cob_match.empty:
        return cob_match.iloc[0], "Tier 1 Match: Exact Hiscox COB"
    naics_match = engine_df[engine_df["NAICS_Description"].str.lower() == input_clean]
    if not naics_match.empty:
        return naics_match.iloc[0], "Tier 1 Match: Exact NAICS Description"
    return None, None

def search_with_diagnostics(input_text):
    input_clean = str(input_text).strip().lower()
    results = []
    tier1_row, tier1_reason = check_tier1_rules(input_clean, engine_df)
    if tier1_row is not None:
        return [{
            "Input_Description": input_text,
            "Hiscox_COB": tier1_row["Hiscox_COB"],
            "Confidence (%)": 100.0,
            "Confidence Level": "High Confidence",
            "Confidence Drivers": tier1_reason,
            "Match_Status": "Confirmed",
            "Appetite": summarize_appetite_logic(tier1_row),
            "PL": tier1_row["PL"],
            "GL": tier1_row["GL"],
            "BOP": tier1_row["BOP"],
            "Cyber": tier1_row["Cyber"]
        }]
    embedding = model.encode([input_text], convert_to_numpy=True)
    D, I = index.search(embedding, 3)
    for rank in range(3):
        idx = I[0][rank]
        row = engine_df.iloc[idx]
        score = 1 / (1 + D[0][rank])
        sim_pct = round(score * 100, 1)
        results.append({
            "Input_Description": input_text,
            "Hiscox_COB": row["Hiscox_COB"],
            "Confidence (%)": sim_pct,
            "Confidence Level": confidence_label(sim_pct),
            "Confidence Drivers": explain_confidence(score, input_text),
            "Match_Status": match_status(sim_pct),
            "Appetite": summarize_appetite_logic(row),
            "PL": row["PL"],
            "GL": row["GL"],
            "BOP": row["BOP"],
            "Cyber": row["Cyber"]
        })
    return results

# === UI ===
search_input = st.text_input("🔍 Enter a business description for diagnostic match:")
if search_input:
    st.markdown("### 🧠 Diagnostic Results")
    results = search_with_diagnostics(search_input)
    for res in results:
        st.markdown(f"**{res['Hiscox_COB']}** — {res['Confidence (%)']}% ({res['Confidence Level']})")
        st.markdown(f"→ *Drivers:* {res['Confidence Drivers']}")
        st.markdown(f"→ *Appetite:* {res['Appetite']}")
        st.markdown("---")
