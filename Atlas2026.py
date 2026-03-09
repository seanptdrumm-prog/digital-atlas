import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Atlas 8.0 Live", layout="wide")

# 1. THE LIVE DATA CONNECTION
@st.cache_data(ttl=600)
def load_live_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Load all tabs
    master = pd.read_csv(base_url + "Hiscox543")
    p1 = pd.read_csv(base_url + "p1")
    p2 = pd.read_csv(base_url + "p2")
    p3 = pd.read_csv(base_url + "p3")
    
    # Rule 1a: Clean NAICS (Strip -2022 etc)
    master['Clean_NAICS'] = master['Full_Industry_Code'].astype(str).str.extract(r'(\d{6})')
    
    return master, pd.concat([p1, p2, p3], ignore_index=True)

# 2. THE AI BRAIN (Atlas 7 Style)
@st.cache_resource
def load_brain(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = (df["Hiscox_COB"].astype(str) + " " + df.get("NAICS_Description", "").astype(str)).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

# --- APP START ---
st.title("🛡️ Hiscox Atlas 8.0")
master_df, partner_df = load_live_data()
model, index = load_brain(master_df)

query = st.text_input("Enter Industry Description or NAICS:")

if query:
    q = query.lower().strip()
    
    # STEP 1: Check Partner Tabs (p1, p2, p3)
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        st.success("✅ Match Found in Partner Approval Tabs")
        st.dataframe(p_match)
    else:
        # STEP 2: Semantic AI Search in Master
        st.info("🔍 Searching Master 543 via AI...")
        vec = model.encode([q], convert_to_numpy=True)
        D, I = index.search(vec, k=3)
        for idx in I[0]:
            match = master_df.iloc[idx]
            st.write(f"**{match['Hiscox_COB']}** (Code: {match['Full_Industry_Code']})")
