import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Hiscox Atlas 2026", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #ff4b4b;
        height: 50px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    file_path = "AtlasEngine2026_cleaned.xlsx"

    master = pd.read_excel(file_path, sheet_name="Hiscox543")
    partner = pd.read_excel(file_path, sheet_name="Partner_Mapping")
    naics = pd.read_excel(file_path, sheet_name="NAICS_Mapping")

    # Clean columns
    master.columns = [clean_text(c) for c in master.columns]
    partner.columns = [clean_text(c) for c in partner.columns]
    naics.columns = [clean_text(c) for c in naics.columns]

    # Clean all text cells
    for df in [master, partner, naics]:
        for col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normalize key master field
    master["COB_NORM"] = master["Hiscox_COB"].apply(norm)

    # Keep only helper rows whose COB actually exists in master
    valid_cobs = set(master["COB_NORM"])

    if "Hiscox_COB" in partner.columns:
        partner = partner[partner["Hiscox_COB"].apply(norm).isin(valid_cobs)].copy()

    if "Hiscox_COB" in naics.columns:
        naics = naics[naics["Hiscox_COB"].apply(norm).isin(valid_cobs)].copy()

    return master, partner, naics

master_df, partner_df, naics_df = load_data()

# =========================
# OVERRIDE RULES
# =========================
def check_overrides(q):
    if any(t in q for t in ["distributor", "distributors", "dealer", "dealers", "wholesale", "wholesaler"]):
        return "Distribution, Wholesalers, Dealers and Other Sales", "Override: distribution/dealer rule"

    if "auto" in q and "repair" in q:
        return "Auto, Car, Truck, Boat", "Override: auto repair rule"

    if any(t in q for t in ["church", "temple", "mosque", "worship"]):
        return "Religious", "Override: religious rule"

    if ("doctor" in q or "doctors" in q) and not any(t in q for t in ["psych", "psychi", "veterin", "dental", "therapy", "clinic"]):
        return "Physicians Office", "Override: doctor rule"

    return None, None

# =========================
# PARTNER MATCH
# =========================
def partner_match(query, detail_query=None, partner_name=None):
    if partner_df.empty:
        return None, None

    df = partner_df.copy()

    # Safe defaults if optional columns exist / don't exist
    for col in ["Partner", "Partner_Description", "Partner_Secondary_Description", "Hiscox_COB"]:
        if col not in df.columns:
            df[col] = ""

    df["_partner"] = df["Partner"].apply(norm)
    df["_desc"] = df["Partner_Description"].apply(norm)
    df["_detail"] = df["Partner_Secondary_Description"].apply(norm)
    df["_cob"] = df["Hiscox_COB"]

    q = norm(query)
    dq = norm(detail_query) if detail_query else ""
    pn = norm(partner_name) if partner_name else ""

    if pn:
        df = df[df["_partner"] == pn]

    # Exact primary + exact detail
    if dq:
        hit = df[(df["_desc"] == q) & (df["_detail"] == dq)]
        if not hit.empty:
            return hit.iloc[0]["Hiscox_COB"], "Partner exact primary + secondary"

    # Exact primary
    hit = df[df["_desc"] == q]
    if not hit.empty:
        return hit.iloc[0]["Hiscox_COB"], "Partner exact primary"

    # Contains primary + detail
    if dq:
        hit = df[
            df["_desc"].str.contains(q, na=False) &
            df["_detail"].str.contains(dq, na=False)
        ]
        if not hit.empty:
            return hit.iloc[0]["Hiscox_COB"], "Partner contains primary + secondary"

    # Contains primary
    hit = df[df["_desc"].str.contains(q, na=False)]
    if not hit.empty:
        return hit.iloc[0]["Hiscox_COB"], "Partner contains primary"

    return None, None

# =========================
# NAICS MATCH
# =========================
def naics_match(query):
    if naics_df.empty:
        return None, None

    desc_col = None
    for c in naics_df.columns:
        cl = c.lower()
        if "description" in cl:
            desc_col = c
            break

    if not desc_col or "Hiscox_COB" not in naics_df.columns:
        return None, None

    df = naics_df.copy()
    df["_desc"] = df[desc_col].apply(norm)

    q = norm(query)

    # Exact
    hit = df[df["_desc"] == q]
    if not hit.empty:
        return hit.iloc[0]["Hiscox_COB"], "Exact NAICS description match"

    # Fuzzy
    choices = df[desc_col].dropna().astype(str).tolist()
    best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)

    if best and best[1] >= 90:
        matched_desc = best[0]
        row = df[df[desc_col] == matched_desc].iloc[0]
        return row["Hiscox_COB"], f"Fuzzy NAICS match ({int(best[1])}%)"

    return None, None

# =========================
# COB FALLBACK
# =========================
def cob_fallback_match(query):
    choices = master_df["Hiscox_COB"].dropna().astype(str).tolist()
    best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)

    if best and best[1] >= 85:
        return best[0], f"Fuzzy COB fallback ({int(best[1])}%)"

    return None, None

# =========================
# FIND FINAL MASTER ROW
# =========================
def find_master_row(cob_name):
    hit = master_df[master_df["COB_NORM"] == norm(cob_name)]
    if not hit.empty:
        return hit.iloc[0]
    return None

# =========================
# UI
# =========================
st.title("Hiscox Atlas 2026")
st.caption("Prototype powered by Hiscox543 + Partner_Mapping + NAICS_Mapping")

partner_name = st.selectbox(
    "Partner (optional)",
    options=[""] + sorted([p for p in partner_df["Partner"].dropna().unique().tolist() if str(p).strip()]),
    index=0
)

query = st.text_input("Search for a business description or 6-digit NAICS code")
detail_query = st.text_input("Secondary description/details (optional, mainly for Talage)")

if query:
    target_cob = None
    match_source = None

    # 1. Override
    target_cob, match_source = check_overrides(norm(query))

    # 2. Partner mapping
    if not target_cob:
        target_cob, match_source = partner_match(
            query=query,
            detail_query=detail_query,
            partner_name=partner_name if partner_name else None
        )

    # 3. NAICS mapping
    if not target_cob:
        target_cob, match_source = naics_match(query)

    # 4. Fuzzy fallback to COB
    if not target_cob:
        target_cob, match_source = cob_fallback_match(query)

    if target_cob:
        row = find_master_row(target_cob)

        if row is not None:
            st.success(f"Match found: {row['Hiscox_COB']}")
            st.write(f"**Match source:** {match_source}")
            st.write(f"**COB Group:** {row.get('COB_Group', '')}")
            st.write(f"**Full Industry Code:** {row.get('Full_Industry_Code', '')}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GL", row.get("GL", ""))
            c2.metric("PL", row.get("PL", ""))
            c3.metric("BOP", row.get("BOP", ""))
            c4.metric("Cyber", row.get("Cyber", ""))

            if row.get("Definition", ""):
                st.info(f"**Definition:** {row.get('Definition', '')}")

            if row.get("State Restrictions", ""):
                st.warning(f"**State Restrictions:** {row.get('State Restrictions', '')}")

            with st.expander("Diagnostics"):
                st.write({
                    "query": query,
                    "secondary_description": detail_query,
                    "partner": partner_name,
                    "match_source": match_source,
                    "target_cob": target_cob
                })
        else:
            st.error(f"Matched '{target_cob}' but could not find it in Hiscox543.")
    else:
        st.error("No match found.")
