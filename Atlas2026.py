import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=600)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    master = pd.read_csv(base_url + "Hiscox543")
    # Clean column names immediately to avoid hidden space issues
    master.columns = [str(c).strip().upper() for c in master.columns]
    
    p_tabs = pd.concat([pd.read_csv(base_url + "p1"), 
                        pd.read_csv(base_url + "p2"), 
                        pd.read_csv(base_url + "p3")], ignore_index=True)
    p_tabs.columns = [str(c).strip().upper() for c in p_tabs.columns]
    
    return master, p_tabs

master_df, partner_df = load_data()

def display_appetite_boxes(row):
    # Try to find the COB name in the first two columns
    cob_name = row.iloc[1] if len(row) > 1 else "Unknown Class"
    st.subheader(f"Results for: {cob_name}")
    
    cols = st.columns(4)
    target_lobs = ["GL", "PL", "BOP", "CYBER"]
    
    for i, lob in enumerate(target_lobs):
        # Look through all column names to find one that matches the LOB
        status = "N/A"
        for col_name in row.index:
            if lob in col_name:
                status = str(row[col_name]).strip().upper()
                break
        
        # Display Logic
        if "Y" in status or "YES" in status:
            cols[i].success(f"### {lob}\n**YES**")
        elif "N" in status or "NO" in status:
            cols[i].error(f"### {lob}\n**NO**")
        else:
            cols[i].info(f"### {lob}\n**{status}**")

# --- UI ---
st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry, NAICS, or Description:")

if query:
    q = query.lower().strip()
    
    # Check both DataFrames
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    m_match = master_df[master_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    if not p_match.empty:
        display_appetite_boxes(p_match.iloc[0])
    elif not m_match.empty:
        display_appetite_boxes(m_match.iloc[0])
    else:
        st.warning("No match found in the Hiscox543 Master List.")
