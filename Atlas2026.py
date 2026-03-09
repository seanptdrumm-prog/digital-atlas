import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hiscox Appetite Atlas 8.0", layout="wide")

@st.cache_data(ttl=600)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    master = pd.read_csv(base_url + "Hiscox543")
    p_tabs = pd.concat([pd.read_csv(base_url + "p1"), 
                        pd.read_csv(base_url + "p2"), 
                        pd.read_csv(base_url + "p3")], ignore_index=True)
    return master, p_tabs

master_df, partner_df = load_data()

def display_appetite_boxes(row):
    # This matches your sheet: Col 3=GL, Col 4=PL, Col 5=BOP, Col 6=Cyber
    st.subheader(f"Results for: {row.iloc[1]}") # The COB Name
    
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
    
    for name, idx, col in lobs:
        # We look specifically at the cell value
        val = str(row.iloc[idx]).strip().upper()
        
        if "Y" in val:
            col.success(f"### {name}\n**YES**")
        else:
            col.error(f"### {name}\n**NO**")

# --- UI ---
st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry or NAICS:")

if query:
    q = query.lower().strip()
    # 1. Check Partners first
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        display_appetite_boxes(p_match.iloc[0])
    else:
        # 2. Check Master
        m_match = master_df[master_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
        if not m_match.empty:
            display_appetite_boxes(m_match.iloc[0])
        else:
            st.warning("No match found. Please check your spelling or try a different keyword.")
