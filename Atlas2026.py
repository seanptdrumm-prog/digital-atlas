import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=300)
def load_all_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Load Master and clean NAICS
    master = pd.read_csv(base_url + "Hiscox543")
    # Identify NAICS column (usually index 6 based on our debugger)
    master['NAICS_CLEAN'] = master.iloc[:, 6].astype(str).str.extract(r'(\d{6})')
    
    # Load and combine Partner tabs
    p1 = pd.read_csv(base_url + "p1")
    p2 = pd.read_csv(base_url + "p2")
    p3 = pd.read_csv(base_url + "p3")
    partners = pd.concat([p1, p2, p3], ignore_index=True)
    
    return master, partners

master_df, partner_df = load_all_data()

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry Name or 6-Digit NAICS Code:")

def display_card(row, source_type):
    st.success(f"✅ Match Found via {source_type}")
    st.subheader(f"Class: {row.iloc[1]}")
    
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
    
    for name, idx, col in lobs:
        val = str(row.iloc[idx]).strip().upper()
        if "Y" in val:
            col.success(f"### {name}\n**YES**")
        else:
            col.error(f"### {name}\n**NO**")
    
    st.info(f"**Definition:** {row.iloc[7]}")

if query:
    q = query.strip().lower()
    
    # 1. SEARCH NAICS (If input is 6 digits)
    if q.isdigit() and len(q) == 6:
        naics_match = master_df[master_df['NAICS_CLEAN'] == q]
        if not naics_match.empty:
            display_card(naics_match.iloc[0], "Exact NAICS Code")
            st.stop()

    # 2. SEARCH PARTNER TABS (p1, p2, p3)
    # This looks for your term in ANY column of the partner sheets
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    if not p_match.empty:
        display_card(p_match.iloc[0], "Partner Approval Mapping")
        st.stop()

    # 3. SEARCH MASTER (Fuzzy Fallback)
    choices = master_df.iloc[:, 1].tolist()
    best_match = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
    
    if best_match and best_match[1] > 70:
        row = master_df[master_df.iloc[:, 1] == best_match[0]].iloc[0]
        display_card(row, f"AI Recommendation ({int(best_match[1])}% Match)")
    else:
        st.warning("No match found. Try a different keyword or a 6-digit NAICS code.")
