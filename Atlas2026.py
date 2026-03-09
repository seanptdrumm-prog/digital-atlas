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

# Small stopgap for extreme abbreviations while we tune
SYNONYMS = {"pr": "public relations", "hr": "human resources", "it": "information technology"}

# === 2. DATA LOADING & UNIFIED INDEX ===
@st.cache_data(ttl=3600)
def load_and_index_data():
    file_path = "AtlasEngine2026_cleaned.xlsx"
    
    master = pd.read_excel(file_path, sheet_name="Hiscox543").fillna("").astype(str)
    partner = pd.read_excel(file_path, sheet_name="Partner_Mapping").fillna("").astype(str)
    naics = pd.read_excel(file_path, sheet_name="NAICS_Mapping").fillna("").astype(str)
    
    search_index = {}
    
    for _, r in master.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        if cob: search_index[cob.lower()] = cob
            
    for _, r in partner.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        desc1 = r.get("Partner_Description", "").strip().lower()
        desc2 = r.get("Partner_Secondary_Description", "").strip().lower()
        if cob:
            if desc1: search_index[desc1] = cob
            if desc2: search_index[desc2] = cob
                
    for _, r in naics.iterrows():
        cob = r.get("Hiscox_COB", "").strip()
        desc = r.get("NAICS_Description", "").strip().lower()
        if cob and desc:
            search_index[desc] = cob

    return master, search_index

master_df, search_index = load_and_index_data()

# === 3. OVERRIDES ===
def check_overrides(q):
    if "auto" in q and "repair" in q: return "Auto, Car, Truck, Boat"
    if "doctor" in q and not any(t in q for t in ["psych", "vet", "dental"]): return "Physicians office"
    return None

# === 4. UI RENDER ENGINE ===
def render_tuning_diagnostics(query_used, match_reason, top_3_results):
    # This box is permanently open and shows the top 3 hits so you can gauge the scoring
    with st.expander("⚙️ Tuning & Diagnostics (Always Visible)", expanded=True):
        st.write(f"**Cleaned Search Term:** `{query_used}`")
        st.write(f"**Match Logic Triggered:** {match_reason}")
        
        if top_3_results:
            st.write("---")
            st.write("**Top 3 Engine Candidates:**")
            for i, res in enumerate(top_3_results):
                phrase, score = res[0], res[1]
                target = search_index.get(phrase, "Unknown")
                st.write(f"{i+1}. `{phrase}` ➡️ Mapped to: **{target}** (Confidence: **{int(score)}%**)")
        else:
            st.write("No fuzzy candidates generated (Exact match or override used).")

def display_verdict(cob_name):
    if cob_name == "OOA":
        st.markdown('<div class="scorecard app-no"><h2>OUT OF APPETITE (OOA)</h2><p>This class is universally restricted.</p></div>', unsafe_allow_html=True)
        return

    rule_row = master_df[master_df["Hiscox_COB"].str.lower() == cob_name.lower()]
    
    if rule_row.empty:
        st.error(f"⚠️ Mapped to '{cob_name}', but it's missing from the Hiscox543 Master List.")
        return
    
    row = rule_row.iloc[0]
    
    gl = "Y" in row.get("GL", "").upper()
    pl = "Y" in row.get("PL", "").upper()
    bop = "Y" in row.get("BOP", "").upper()
    cyb = "Y" in row.get("Cyber", "").upper()
    
    yes_flags = [f for f, b in zip(["GL", "PL", "BOP", "Cyber"], [gl, pl, bop, cyb]) if b]
    
    if len(yes_flags) >= 2:
        status, css_class = "IN APPETITE", "app-yes"
    elif len(yes_flags) == 1:
        status, css_class = f"LIMITED APPETITE ({yes_flags[0]} ONLY)", "app-yes"
    else:
        status, css_class = "OUT OF APPETITE", "app-no"
    
    st.markdown(f'<div class="scorecard {css_class}"><h2>{status}</h2><p>Hiscox Class: <b>{row.get("Hiscox_COB", "")}</b></p></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    for name, is_yes, col in zip(["GL", "PL", "BOP", "Cyber"], [gl, pl, bop, cyb], [c1, c2, c3, c4]):
        col.markdown(f'<div class="lob-box {"lob-yes" if is_yes else "lob-no"}">{name}<br>{"YES" if is_yes else "NO"}</div>', unsafe_allow_html=True)

    defn = row.get("Definition", "").strip()
    st.markdown("---")
    st.info(f"**📖 Definition:** {defn if defn else 'No specific definition provided.'}")

# === 5. SEARCH ENGINE ===
st.title("🛡️ Atlas 2026: Underwriting Engine")
query = st.text_input("Enter business description, keyword, or industry:")

if query:
    q = query.lower().strip()
    q_clean = " ".join([SYNONYMS.get(w, w) for w in q.split()])
    
    target_cob = None
    match_source = ""
    top_candidates = []

    override_hit = check_overrides(q_clean)
    
    if override_hit:
        target_cob, match_source = override_hit, "Override Rule"
    elif q_clean in search_index:
        target_cob, match_source = search_index[q_clean], "Exact Term Match"
    else:
        choices = list(search_index.keys())
        # Calculate the top 3 closest matches instead of just 1
        top_candidates = process.extract(q_clean, choices, scorer=fuzz.token_set_ratio, limit=3)
        
        best_match = top_candidates[0] if top_candidates else None
        
        # Currently set to 78% threshold
        if best_match and best_match[1] >= 78:
            target_cob = search_index[best_match[0]]
            match_source = "Fuzzy Match"

    if target_cob:
        display_verdict(target_cob)
    else:
        st.error("❌ No match found. See diagnostics below to see what the engine was considering.")
        
    # ALWAYS render the tuning box at the bottom
    render_tuning_diagnostics(q_clean, match_source if target_cob else "Failed to meet 78% threshold", top_candidates)
