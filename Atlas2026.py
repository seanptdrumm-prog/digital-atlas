import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=300)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Hiscox543"
    return pd.read_csv(url)

df = load_data()

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry (e.g., 'Yoga', 'PR Firm', 'Architect'):")

if query:
    q = query.lower().strip()
    
    # STAGE 1: Exact Keyword Match (Highest Priority)
    # We look for the word specifically in the COB name or Description
    mask = df.iloc[:, 1].str.lower().str.contains(q, na=False) | \
           df.iloc[:, 7].str.lower().str.contains(q, na=False)
    
    exact_matches = df[mask]
    
    if not exact_matches.empty:
        # If we found an exact word match, take the best one
        row = exact_matches.iloc[0]
        st.success(f"🎯 Exact Match Found")
    else:
        # STAGE 2: Smart Fuzzy Match (Fallback)
        choices = df.iloc[:, 1].tolist()
        best_match = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
        
        if best_match and best_match[1] > 70:
            row = df[df.iloc[:, 1] == best_match[0]].iloc[0]
            st.info(f"🔍 Best AI Recommendation (Confidence: {int(best_match[1])}%)")
        else:
            row = None

    if row is not None:
        st.subheader(f"Results for: {row.iloc[1]}")
        
        # APPETITE BOXES
        c1, c2, c3, c4 = st.columns(4)
        lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
        
        for name, idx, col in lobs:
            val = str(row.iloc[idx]).strip().upper()
            if "YES" in val or "Y" in val:
                col.success(f"### {name}\n**YES**")
            else:
                col.error(f"### {name}\n**NO**")
        
        st.write(f"**Definition:** {row.iloc[7]}")
    else:
        st.warning("No match found. Try a more specific keyword like 'Yoga' or 'Architect'.")
