import streamlit as st
import pandas as pd

st.set_page_config(page_title="Atlas 8.0 Debugger", layout="wide")

@st.cache_data(ttl=1) # NO CACHING - Live data only
def load_debug_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    master = pd.read_csv(base_url + "Hiscox543")
    return master

df = load_debug_data()

st.title("🛡️ Atlas 8.0 Debugger")
query = st.text_input("Type 'Consulting' here:")

if query:
    # 1. Show the result we found
    match = df[df.apply(lambda r: query.lower() in str(r.values).lower(), axis=1)]
    
    if not match.empty:
        st.write("### 🔍 What the App Found:")
        st.dataframe(match.head(1)) # This shows the RAW row exactly as Python sees it
        
        st.write("### 🧪 Column Header Test:")
        st.write(list(df.columns)) # This lists every header Python detected
    else:
        st.error("The App literally cannot find that word in your Google Sheet.")
