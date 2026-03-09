import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# === 1. THEME & SETUP ===
st.set_page_config(page_title="Hiscox Atlas 2026", layout="wide")

st.markdown("""
    <style>
        body, .stApp { background-color: #000000; color: #FFFFFF; }
        .stTextInput > div > div > input { background-color: #222222; color: #ffffff; border: 1px solid #ff4b4b; height: 50px; font-size: 20px; }
        .scorecard { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px; border: 2px solid #333; }
        .app-yes { background-color: #00cc66; color: black; box-shadow: 0 0 15px rgba(0,204,102,0.4); }
        .app-no { background-color: #cc3333; color: white; box-shadow: 0 0 15px rgba(204,51,51,0.4); }
        .lob-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin: 5px; }
        .lob-yes { background-color: #004d26; color: #00ff88; border: 1px solid #00ff88; }
        .lob-no { background-color: #4d0000; color: #ff4b4b; border: 1px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# === 2. DATA LOADING & SCRUBBING ===
@st.cache_data(ttl=120)
def load_and_scrub_data():
    sheet_id = "1XiT2GVCwdM2_F2-MHQOVVy-ZMAAL5_Tz0AaIkdPs-U0"
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    def fetch_clean(sheet_name):
        df = pd.read_csv(base_url + sheet_name)
        df = df.fillna("").astype(str)
        df.columns = df.columns.str.strip().str.upper()
        return df

    master = fetch_clean("Hiscox543")
    p1 = fetch_clean("p1")
    p2 = fetch_clean("p2")
    p3 = fetch_clean("p3")
    
    # Keeping tabs separate for diagnostic clarity
    return master, p1, p2, p3

master_df, p1_df, p2_df, p3_df = load_and_scrub_data()
partner_df = pd.concat([p1_df, p2_df, p3_df], ignore_index=True)

# === 3. OVERRIDE GUARDRAILS ===
def check_overrides(q):
    if any(t in q for t in ["distributor", "dealer", "wholesale"]):
        return "Distribution, Wholesalers, Dealers"
    if "auto" in q and "repair" in q:
        return "Auto, Car, Truck, Boat"
    if "doctor" in q and not any(t in q for t in ["psych", "vet", "dental"]):
        return "Physicians office"
    if any(t in q for t in ["church", "temple", "mosque"]):
        return "Religious"
    return None

# === 4. UI RENDER ENGINE ===
def display_rulebook_verdict(cob_name, match_type, raw_bridge_match=None):
    rule_row = master_df[master_df.iloc[:, 1].str.strip().str.lower() == cob_name.lower()]
    
    if rule_row.empty:
        st.error(f"⚠️ App resolved to Class: '{cob_name}', but it cannot find this exact text in Column B of the Hiscox543 Master List.")
        
        # Diagnostic Box for Failures
        with st.expander("🔍 Match Diagnostics (Why this failed)", expanded=True):
            st.write(f"**1. Search Logic Triggered:** {match_type}")
            st.write(f"**2. Target Class Attempted:** `{cob_name}`")
            if raw_bridge_match is not None:
                st.write("**3. Raw Data Pulled from Partner Tab:**")
                st.dataframe(raw_bridge_match)
        return
    
    row = rule_row.iloc[0]
    
    gl = "Y" in row.iloc[2].upper()
    pl = "Y" in row.iloc[3].upper()
    bop = "Y" in row.iloc[4].upper()
    cyb = "Y" in row.iloc[5].upper()
    
    yes_count = sum([gl, pl, bop, cyb])
    status = "IN APPETITE" if yes_count >= 2 else "OUT OF APPETITE"
    css_class = "app-yes" if status == "IN APPETITE" else "app-no"
    
    st.markdown(f'<div class="scorecard {css_class}"><h2>{status}</h2><p>Class: <b>{row.iloc[1]}</b></p></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", gl, c1), ("PL", pl, c2), ("BOP", bop, c3), ("Cyber", cyb, c4)]
    
    for name, is_yes, col in lobs:
        if is_yes:
            col.markdown(f'<div class="lob-box lob-yes">{name}<br>YES</div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="lob-box lob-no">{name}<br>NO</div>', unsafe_allow_html=True)

    defn = row.iloc[7].strip()
    restr = row.iloc[8].strip() if len(row) > 8 else ""
    
    st.markdown("---")
    st.info(f"**📖 Definition:** {defn if defn else 'No definition provided in sheet.'}")
    if restr:
        st.warning(f"**🚧 Restrictions:** {restr}")

    # === NEW: DIAGNOSTIC BOX ===
    with st.expander("🔍 Match Diagnostics (How the engine found this)"):
        st.write(f"**1. Search Logic Triggered:** {match_type}")
        st.write(f"**2. Target Class Pulled:** `{cob_name}`")
        if raw_bridge_match is not None:
            st.write("**3. Raw Row Found in Partner Tab (Bridge):**")
            st.dataframe(raw_bridge_match)
        st.write("**4. Raw Row Found in Master Hiscox543 Tab (Rulebook):**")
        st.dataframe(rule_row)

# === 5. THE WATERFALL SEARCH ENGINE ===
st.title("🛡️ Atlas 2026: Underwriting Engine")
query = st.text_input("Enter NAICS description, industry, or keyword:")

if query:
    q = query.lower().strip()
    target_cob = None
    match_source = ""
    raw_bridge_data = None

    override_hit = check_overrides(q)
    if override_hit:
        target_cob = override_hit
        match_source = "Tier 1: Underwriting Override Rule"

    elif any(master_df.iloc[:, 1].str.strip().str.lower() == q):
        target_cob = master_df[master_df.iloc[:, 1].str.strip().str.lower() == q].iloc[0, 1]
        match_source = "Tier 2: Exact Class Name Match (Master)"

    else:
        bridge_match = partner_df[partner_df.apply(lambda r: q in str(r.values).lower(), axis=1)]
        if not bridge_match.empty:
            # Capturing the raw bridge data for the diagnostic box
            raw_bridge_data = bridge_match.iloc[[0]] 
            target_cob = bridge_match.iloc[0, 1] 
            match_source = "Tier 3: NAICS Mapping (Partner Tabs)"
        
        else:
            choices = master_df.iloc[:, 1].tolist()
            best_match = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
            
            if best_match and best_match[1] >= 85:
                target_cob = best_match[0]
                match_source = f"Tier 4: Smart AI Search ({int(best_match[1])}% Confidence)"

    if target_cob:
        display_rulebook_verdict(target_cob, match_source, raw_bridge_data)
    else:
        st.error("❌ No match found.")
