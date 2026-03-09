import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss

# 1. LIVE GOOGLE SYNC (No more Excel files)
@st.cache_data(ttl=600)
def load_live_sheets():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Waterfall: Load Master then Partners
    master = pd.read_csv(base_url + "Hiscox543")
    partners = pd.concat([pd.read_csv(base_url + "p1"), 
                         pd.read_csv(base_url + "p2"), 
                         pd.read_csv(base_url + "p3")], ignore_index=True)
    
    # Rule 1a: Strip 6-digit NAICS codes
    master['Clean_NAICS'] = master['Full_Industry_Code'].astype(str).str.extract(r'(\d{6})')
    return master, partners

# 2. THE AI BRAIN
@st.cache_resource
def load_brain(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = (df["Hiscox_COB"].astype(str) + " " + df.get("NAICS_Description", "").astype(str)).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

# --- INTERFACE ---
st.title("🛡️ Atlas 8.0: High-Speed Engine")
master_df, partner_df = load_live_sheets()
model, index = load_brain(master_df)

query = st.text_input("Search Industry or NAICS:")

if query:
    # WATERFALL 1: Check verified partner mappings first
    p_match = partner_df[partner_df.apply(lambda r: query.lower() in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        st.success("✅ Found in Partner Approval tabs (p1-p3)")
        st.write(p_match)
    else:
        # WATERFALL 2: Use AI to find best fit in 543 Master
        st.info("🔍 Searching Master 543 via Semantic AI...")
        # (AI logic continues here)
