
import pandas as pd
import streamlit as st
import numpy as np
import faiss
import time
import threading
from sentence_transformers import SentenceTransformer

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

# === RULE-BASED MATCHING ===
def is_instructional(description):
    terms = ["instructor", "teacher", "training", "coach", "lesson"]
    return any(term in description for term in terms)

def is_psychology_clinical(description):
    terms = ["psychology", "psychologist", "therapist"]
    context = ["office", "clinic", "center"]
    return any(t in description for t in terms) and any(c in description for c in context)

def is_consulting_term(description):
    return "consultant" in description or "consulting" in description

def naics_code_search(naics_code):
    df = engine_df[engine_df["NAICS_Code"].astype(str).str.startswith(str(naics_code))]
    return df

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

def run_batch_match(inputs):
    results = []
    embeddings = model.encode(inputs, convert_to_numpy=True)
    D, I = index.search(embeddings, 1)
    for i, entry in enumerate(inputs):
        desc = str(entry).strip().lower()
        idx = I[i][0]
        row = engine_df.iloc[idx]
        score = 1 / (1 + D[i][0])
        pct = round(score * 100, 1)

        # === RULE OVERRIDES ===
        if is_instructional(desc) and "therapy" not in desc:
            inst_matches = engine_df[engine_df["Hiscox_COB"].str.contains("Instructor", case=False)]
            if not inst_matches.empty:
                row = inst_matches.iloc[0]
                pct = 92.0

        elif is_psychology_clinical(desc):
            psych_matches = engine_df[engine_df["Hiscox_COB"].str.contains("Psychologist", case=False)]
            if not psych_matches.empty:
                row = psych_matches.iloc[0]
                pct = 93.5

        elif is_consulting_term(desc):
            consult_matches = engine_df[engine_df["Hiscox_COB"].str.contains("Consult", case=False)]
            if not consult_matches.empty:
                row = consult_matches.iloc[0]
                pct = max(pct, 90.0)

        elif "cake" in desc or "bakery" in desc:
            cake_matches = engine_df[engine_df["Hiscox_COB"].str.contains("Bakery|Cake", case=False, regex=True)]
            if not cake_matches.empty:
                row = cake_matches.iloc[0]
                pct = 91.0

        result = {
            "Input_Description": entry,
            "Hiscox_COB": row["Hiscox_COB"],
            "full_industry_code": row.get("full_industry_code", ""),
            "Confidence (%)": confidence_label(pct),
            "Match_Status": match_status(pct),
            "Appetite": summarize_appetite_logic(row),
            "PL": row["PL"],
            "GL": row["GL"],
            "BOP": row["BOP"],
            "Cyber": row["Cyber"]
        }
        results.append(result)
    return pd.DataFrame(results)
