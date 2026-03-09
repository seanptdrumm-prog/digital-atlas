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

# === 2. DATA LOADING & UNIFIED INDEX ===
@st.cache_data(ttl=3600)
def load_and_index_data():
    file_path = "AtlasEngine2026_cleaned.xlsx"
    
    # Load your cleaned local file
    master = pd.read_excel(file_path, sheet_name="Hiscox543").fillna("").astype(str)
    partner = pd.read_excel(file_path, sheet_name="Partner_Mapping").fillna("").astype(str)
    naics = pd.read_excel(file_path, sheet_name="NAICS_Mapping").fillna("").astype(str)
    
    # Build a massive, flat search dictionary: { "Searchable Phrase": "Hiscox_COB" }
    search_index = {}
    
    # 1. Add Master COBs
    for _, r in master.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        if cob: search_index[cob.lower()] = cob
            
    # 2. Add Partner Descriptions
    for _, r in partner.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        desc1 = r.get("Partner_Description", "").strip().lower()
        desc2 = r.get("Partner_Secondary_Description", "").strip().lower()
        if cob:
            if desc1: search_index[desc1] = cob
            if desc2: search_index[desc2] = cob
                
    # 3. Add NAICS Descriptions
    for _, r in naics.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        desc = r.get("NAICS_Description", "").strip().lower()
        if cob and desc:
            search_index[desc] = cob

    return master, search_index

master_df, search_index = load_and_index_data()

# === 3. OVERRIDES ===
def check_overrides(q):
    if any(t in q for t in ["distributor", "dealer", "wholesale"]): return "Distribution, Wholesalers, Dealers"
    if "auto" in q and "repair" in q: return "Auto, Car, Truck, Boat"
    if "doctor" in q and not any(t in q for t in ["psych", "vet", "dental"]): return "Physicians office"
    if any(t in q for t in ["church", "temple", "mosque"]): return "Religious"
    return None

# === 4. UI RENDER ENGINE ===
def display_verdict(cob_name, match_reason, match_score):
    if cob_name == "OOA":
        st.markdown('<div class="scorecard app-no"><h2>OUT OF APPETITE (OOA)</h2><p>This class is universally restricted.</p></div>', unsafe_allow_html=True)
        return

    # Look up by Column Name, not index position!
    rule_row = master_df[master_df["Hiscox_COB"].str.lower() == cob_name.lower()]
    
    if rule_row.empty:
        st.error(f"⚠️ Mapped to '{cob_name}', but it's missing from the Hiscox543 Master List.")
        return
    
    row = rule_row.iloc[0]
    
    gl = "Y" in row.get("GL", "").upper()
    pl = "Y" in row.get("PL", "").upper()
    bop = "Y" in row.get("BOP", "").upper()
    cyb = "Y" in row.get("Cyber", "").upper()
    
    yes_count = sum([gl, pl, bop, cyb])
    status = "IN APPETITE" if yes_count >= 2 else "OUT OF APPETITE"
    css_class = "app-yes" if status == "IN APPETITE" else "app-no"
    
    st.markdown(f'<div class="scorecard {css_class}"><h2>{status}</h2><p>Hiscox Class: <b>{row.get("Hiscox_COB", "")}</b></p></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    lobs = [("GL", gl, c1), ("PL", pl, c2), ("BOP", bop, c3), ("Cyber", cyb, c4)]
    
    for name, is_yes, col in lobs:
        if is_yes:
            col.markdown(f'<div class="lob-box lob-yes">{name}<br>YES</div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="lob-box lob-no">{name}<br>NO</div>', unsafe_allow_html=True)

    defn = row.get("Definition", "").strip()
    restr = row.get("State Restrictions", "").strip()
    
    st.markdown("---")
    st.info(f"**📖 Definition:** {defn if defn else 'No specific definition provided.'}")
    if restr:
        st.warning(f"**🚧 Restrictions:** {restr}")

    # Diagnostics Box (Cleaned up, no partner details shown, just the mapping path)
    with st.expander("🔍 Diagnostics (How we matched this)"):
        st.write(f"**Logic Triggered:** {match_reason}")
        if match_score:
            st.write(f"**Confidence Score:** {match_score}%")
        st.write(f"**Target COB:** `{cob_name}`")

# === 5. SEARCH ENGINE ===
st.title("🛡️ Atlas 2026: Underwriting Engine")
query = st.text_input("Enter business description, keyword, or industry:", placeholder="e.g., Landscaping, PR Firm, Yoga...")

if query:
    q = query.lower().strip()
    target_cob = None
    match_source = ""
    score = None

    override_hit = check_overrides(q)
    
    if override_hit:
        target_cob, match_source = override_hit, "Underwriting Override Rule"
    elif q in search_index:
        # EXACT MATCH: The phrase you typed exactly matches a known description
        target_cob = search_index[q]
        match_source = "Exact Term Match"
        score = 100
    else:
        # FUZZY MATCH: Searches all 20,000+ descriptions simultaneously
        choices = list(search_index.keys())
        
        # token_set_ratio is very forgiving. "landscaping" will 100% match "landscaping services"
        best_match = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
        
        if best_match and best_match[1] >= 75:
            matched_phrase = best_match[0]
            target_cob = search_index[matched_phrase]
            match_source = f"Fuzzy Match (Mapped via '{matched_phrase}')"
            score = int(best_match[1])

    if target_cob:
        display_verdict(target_cob, match_source, score)
    else:
        st.error("❌ No match found. Please refine your keyword.")
