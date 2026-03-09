import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hiscox Atlas 8.0", layout="wide")

@st.cache_data(ttl=300)
def load_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Load Master and Partners
    master = pd.read_csv(base_url + "Hiscox543")
    p1 = pd.read_csv(base_url + "p1")
    p2 = pd.read_csv(base_url + "p2")
    p3 = pd.read_csv(base_url + "p3")
    partners = pd.concat([p1, p2, p3], ignore_index=True)
    
    return master, partners

master_df, partner_df = load_data()

st.title("🛡️ Hiscox Appetite Atlas 8.0")
query = st.text_input("Search Industry or NAICS (e.g., 'Poster', 'Yoga', '541611'):")

def display_result(row):
    # Based on your Debugger: Col 1 is the Class name
    st.subheader(f"Class: {row.iloc[1]}")
    
    # Appetite Grid: Col 2=GL, 3=PL, 4=BOP, 5=Cyber
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", 2, c1), ("PL", 3, c2), ("BOP", 4, c3), ("Cyber", 5, c4)]
    
    for name, idx, col in lobs:
        val = str(row.iloc[idx]).strip().upper()
        if "YES" in val or "Y" in val:
            col.success(f"### {name}\n**IN APPETITE** ✅")
        else:
            col.error(f"### {name}\n**OUT OF APPETITE** ❌")
    
    # Definition: Column 7
    # State Restrictions: Column 8
    st.info(f"**Definition:** {row.iloc[7]}")
    st.warning(f"**Restrictions:** {row.iloc[8]}")

if query:
    q = query.lower().strip()

    # 1. SEARCH PARTNER TABS (Exact match for aliases)
    p_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
    
    # 2. SEARCH MASTER COB NAME (Highest priority for 'Poster', 'Yoga', etc.)
    m_name_match = master_df[master_df.iloc[:, 1].str.lower().str.contains(q, na=False)]
    
    # 3. SEARCH MASTER DESCRIPTION (Fallback)
    m_desc_match = master_df[master_df.iloc[:, 7].str.lower().str.contains(q, na=False)]

    if not p_match.empty:
        display_result(p_match.iloc[0])
    elif not m_name_match.empty:
        display_result(m_name_match.iloc[0])
    elif not m_desc_match.empty:
        display_result(m_desc_match.iloc[0])
    else:
        st.warning(f"No results for '{query}'. Try a simpler keyword.")
