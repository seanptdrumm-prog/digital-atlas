import pandas as pd
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import re

# === CONFIG & THEME ===
st.set_page_config(page_title="Atlas 2026", layout="wide")

st.markdown("""
    <style>
        body, .stApp { background-color: #000000; color: #FFFFFF; }
        .stTextInput > div > div > input { background-color: #222222; color: #ffffff; }
        .success-box { background-color: #00cc66; color: black; padding: 20px; border-radius: 10px; text-align: center; }
        .error-box { background-color: #cc3333; color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .info-box { background-color: #1a1a1a; border-left: 5px solid #ff4b4b; padding: 15px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# === DATA LOADING ===
@st.cache_data(ttl=600)
def load_all_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Master List
    master = pd.read_csv(base_url + "Hiscox543")
    
    # Partner Tabs (p1, p2, p3)
    p_tabs = pd.concat([
        pd.read_csv(base_url + "p1"),
        pd.read_csv(base_url + "p2"),
        pd.read_csv(base_url + "p3")
    ], ignore_index=True)
    
    # Clean Corpus for AI
    master["match_corpus"] = (
        master.iloc[:, 1].fillna("") + " " + 
        master.iloc[:, 7].fillna("") + " " + 
        master.iloc[:, 6].astype(str)
    )
    
    return master, p_tabs

master_df, partner_df = load_all_data()

@st.cache_resource
def build_brain(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

model, index = build_brain(master_df["match_corpus"])

# === APPETITE LOGIC ===
def get_appetite_status(row):
    # Mapping based on verified indices: 2=GL, 3=PL, 4=BOP, 5=Cyber
    flags = [str(row.iloc[2]).strip().lower(), 
             str(row.iloc[3]).strip().lower(), 
             str(row.iloc[4]).strip().lower(), 
             str(row.iloc[5]).strip().lower()]
    
    yes_count = sum(1 for f in flags if f.startswith('y'))
    
    if yes_count >= 2:
        return "IN APPETITE", "success-box"
    elif yes_count == 1:
        return "LIMITED APPETITE", "info-box"
    else:
        return "OUT OF APPETITE", "error-box"

# === SEARCH ENGINE ===
st.title("🛡️ Hiscox Atlas 2026")
query = st.text_input("Search Industry, NAICS, or Description:")

if query:
    q = query.lower().strip()
    
    # TIER 1: Exact Partner Match
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        row = p_match.iloc[0]
        status, css = get_appetite_status(row)
        st.markdown(f'<div class="{css}"><h2>{status}</h2></div>', unsafe_allow_html=True)
        st.subheader(f"Class: {row.iloc[1]}")
    else:
        # TIER 2: AI Search
        emb = model.encode([q], convert_to_numpy=True)
        D, I = index.search(emb, 1)
        row = master_df.iloc[I[0][0]]
        
        status, css = get_appetite_status(row)
        st.markdown(f'<div class="{css}"><h2>{status}</h2></div>', unsafe_allow_html=True)
        st.subheader(f"AI Recommended: {row.iloc[1]}")

    # DISPLAY LOB GRID
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
    
    for name, idx, col in lobs:
        val = str(row.iloc[idx]).strip().upper()
        if "Y" in val:
            col.success(f"**{name}**\nYES")
        else:
            col.error(f"**{name}**\nNO")

    st.markdown(f"**Description:** {row.iloc[7]}")
