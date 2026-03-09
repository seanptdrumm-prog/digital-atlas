import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

# 1. DATA LOADING
@st.cache_data(ttl=600)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    master = pd.read_csv(base_url + "Hiscox543")
    # Combine partners
    p_tabs = pd.concat([pd.read_csv(base_url + "p1"), 
                        pd.read_csv(base_url + "p2"), 
                        pd.read_csv(base_url + "p3")], ignore_index=True)
    return master, p_tabs

# 2. AI BRAIN
@st.cache_resource
def load_ai(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Combines COB and Description for searching
    corpus = (df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

master_df, partner_df = load_data()
model, index = load_ai(master_df)

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry, NAICS, or Description:")

def display_appetite_card(row):
    """Creates a clean visual card for Appetite status"""
    st.subheader(f"Class: {row.get('Hiscox_COB', 'Unknown')}")
    
    # Create 4 columns for the LOBs
    c1, c2, c3, c4 = st.columns(4)
    
    # Dynamic check for Appetite Columns
    lobs = {"GL": c1, "PL": c2, "BOP": c3, "Cyber": c4}
    
    for lob, col in lobs.items():
        # Look for a column in the spreadsheet that matches the LOB name
        status = "N/A"
        for col_name in row.index:
            if lob.lower() in col_name.lower():
                status = str(row[col_name]).strip().upper()
                break
        
        # Style based on Y/N
        if "Y" in status:
            col.success(f"**{lob}**: In Appetite ✅")
        elif "N" in status:
            col.error(f"**{lob}**: Out of Appetite ❌")
        else:
            col.info(f"**{lob}**: {status}")

if query:
    # WATERFALL 1: Partner Check
    p_match = partner_df[partner_df.apply(lambda r: query.lower() in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        st.markdown("### ✅ Partner Approved Mapping Found")
        for _, row in p_match.head(1).iterrows():
            display_appetite_card(row)
    else:
        # WATERFALL 2: AI Search
        st.markdown("### 🔍 AI Recommended Match (Master 543)")
        vec = model.encode([query], convert_to_numpy=True)
        D, I = index.search(vec, k=1)
        display_appetite_card(master_df.iloc[I[0][0]])
