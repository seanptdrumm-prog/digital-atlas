import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=300)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Load Master
    master = pd.read_csv(base_url + "Hiscox543")
    
    # Load Partners (p1, p2, p3)
    p1 = pd.read_csv(base_url + "p1")
    p2 = pd.read_csv(base_url + "p2")
    p3 = pd.read_csv(base_url + "p3")
    partners = pd.concat([p1, p2, p3], ignore_index=True)
    
    return master, partners

master_df, partner_df = load_data()

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry, NAICS, or Keyword (e.g., 'Poster', 'Yoga'):")

def display_result(row, source):
    st.success(f"✅ Result found via {source}")
    st.subheader(f"Class: {row.iloc[1]}") # Hiscox_COB
    
    # Appetite Boxes
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
    
    for name, idx, col in lobs:
        val = str(row.iloc[idx]).strip().upper()
        if "Y" in val:
            col.success(f"### {name}\n**YES**")
        else:
            col.error(f"### {name}\n**NO**")
    
    # Definition - We scan columns 7, 8, and 9 to find the first one that isn't empty
    defn = "No definition provided in spreadsheet."
    for i in [7, 8, 9]:
        if i < len(row) and pd.notna(row.iloc[i]) and str(row.iloc[i]).strip().lower() != "nan":
            defn = row.iloc[i]
            break
    st.info(f"**Definition:** {defn}")

if query:
    q = query.lower().strip()

    # 1. THE "COMMON SENSE" KEYWORD SEARCH (Highest Priority)
    # Checks if the word you typed is actually IN the Class Name or Partner Terms
    p_exact = partner_df[partner_df.iloc[:, 1].str.lower().str.contains(q, na=False)]
    m_exact = master_df[master_df.iloc[:, 1].str.lower().str.contains(q, na=False)]
    
    if not p_exact.empty:
        display_result(p_exact.iloc[0], "Partner Approvals")
    elif not m_exact.empty:
        display_result(m_exact.iloc[0], "Master List")
    else:
        # 2. THE DESCRIPTION SEARCH (Secondary)
        # If it's not in the title, look in the definition column
        m_desc = master_df[master_df.iloc[:, 7].str.lower().str.contains(q, na=False)]
        if not m_desc.empty:
            display_result(m_desc.iloc[0], "Industry Description Match")
        else:
            st.warning(f"No clear match for '{query}'. Try a simpler keyword.")
