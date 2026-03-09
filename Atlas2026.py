import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Atlas 8.0 (Flexible)", layout="wide")

# 1. SMART DATA LOADING (Finds columns regardless of names)
@st.cache_data(ttl=600)
def load_live_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    def fetch_and_clean(name):
        df = pd.read_csv(base_url + name)
        # Standardize: lowercase everything and remove spaces for the engine's internal use
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df

    master = fetch_and_clean("Hiscox543")
    p1 = fetch_and_clean("p1")
    p2 = fetch_and_clean("p2")
    p3 = fetch_and_clean("p3")
    
    # Identify the NAICS column (usually contains 'code' or 'naics')
    naics_cols = [c for c in master.columns if 'code' in c or 'naics' in c]
    if naics_cols:
        master['clean_naics'] = master[naics_cols[0]].astype(str).str.extract(r'(\d{6})')
    
    return master, pd.concat([p1, p2, p3], ignore_index=True)

# 2. DYNAMIC AI BRAIN
@st.cache_resource
def build_brain(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Find the COB and Description columns by keywords
    cob_col = [c for c in df.columns if 'cob' in c or 'class' in c][0]
    desc_col = [c for c in df.columns if 'desc' in c or 'industry' in c][0]
    
    corpus = (df[cob_col].astype(str) + " " + df[desc_col].astype(str)).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, cob_col

# --- INTERFACE ---
st.title("🛡️ Hiscox Atlas 8.0 (Flexible Engine)")
master_df, partner_df = load_live_data()
model, index, active_cob_column = build_brain(master_df)

query = st.text_input("Enter Industry, Description, or NAICS:")

if query:
    q = query.lower().strip()
    
    # WATERFALL 1: Check Partners first
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        st.success("✅ Found in Partner Approval Tabs")
        st.dataframe(p_match)
    else:
        # WATERFALL 2: AI Semantic Search
        st.info("🔍 Searching Master 543 via AI...")
        vec = model.encode([q], convert_to_numpy=True)
        D, I = index.search(vec, k=3)
        for idx in I[0]:
            match = master_df.iloc[idx]
            st.write(f"**{match[active_cob_column].upper()}**")
