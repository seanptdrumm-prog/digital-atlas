import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=300)
def load_final_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Hiscox543"
    df = pd.read_csv(url)
    return df

df = load_final_data()

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry (e.g., 'Consultant'):")

if query:
    # FUZZY SEARCH: Finds 'consulting' even if you type 'consultant'
    choices = df.iloc[:, 1].tolist() # Search the Hiscox_COB column
    best_match = process.extractOne(query, choices, scorer=fuzz.WRatio)
    
    if best_match and best_match[1] > 60: # 60% confidence threshold
        match_text = best_match[0]
        row = df[df.iloc[:, 1] == match_text].iloc[0]
        
        st.subheader(f"Results for: {row.iloc[1]}")
        
        # APPETITE BOXES (Using the 2, 3, 4, 5 map from our Debugger)
        c1, c2, c3, c4 = st.columns(4)
        lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
        
        for name, idx, col in lobs:
            val = str(row.iloc[idx]).strip().upper()
            if "YES" in val or "Y" in val:
                col.success(f"### {name}\n**YES**")
            else:
                col.error(f"### {name}\n**NO**")
        
        # DEFINITION
        st.info(f"**Definition:** {row.iloc[7]}")
    else:
        st.warning("Could not find a close match. Try a different keyword.")
