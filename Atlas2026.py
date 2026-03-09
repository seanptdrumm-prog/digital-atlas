import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Hiscox Atlas 2026", layout="wide")

st.markdown("""
<style>
    body, .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #ff4b4b;
        height: 52px;
        font-size: 20px;
        border-radius: 8px;
    }

    .scorecard {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid #333;
    }

    .app-yes {
        background-color: #00cc66;
        color: black;
        box-shadow: 0 0 15px rgba(0,204,102,0.35);
    }

    .app-no {
        background-color: #cc3333;
        color: white;
        box-shadow: 0 0 15px rgba(204,51,51,0.35);
    }

    .lob-box {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 5px;
    }

    .lob-yes {
        background-color: #004d26;
        color: #00ff88;
        border: 1px solid #00ff88;
    }

    .lob-no {
        background-color: #4d0000;
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
    }

    .small-note {
        font-size: 0.9rem;
        color: #cccccc;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

if "batch_review_rows" not in st.session_state:
    st.session_state.batch_review_rows = []

if "batch_review_filename" not in st.session_state:
    st.session_state.batch_review_filename = ""

# =========================
# SMALL SYNONYM BRIDGE
# =========================
SYNONYMS = {
    "pr": "public relations",
    "hr": "human resources",
    "it": "information technology",
    "cpa": "certified public accountant",
    "cpas": "certified public accountants",
    "ad": "advertising",
}

# =========================
# TEXT HELPERS
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())

def norm(x):
    text = clean_text(x).lower()
    text = re.sub(r"[^a-z0-9&/\-+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_synonyms(text):
    words = norm(text).split()
    return " ".join(SYNONYMS.get(w, w) for w in words)

def normalize_yes_no(x):
    v = norm(x)
    return v in {"y", "yes", "true", "1"}

def looks_like_active_yes(x):
    v = norm(x)
    return v in {"y", "yes", "true", "1", "active"}

# =========================
# DATA LOAD
# =========================
@st.cache_data(ttl=3600)
def load_data():
    file_path = "AtlasEngine2026_cleaned.xlsx"

    master = pd.read_excel(file_path, sheet_name="Hiscox543").fillna("")
    partner = pd.read_excel(file_path, sheet_name="Partner_Mapping").fillna("")
    naics = pd.read_excel(file_path, sheet_name="NAICS_Mapping").fillna("")

    master.columns = [clean_text(c) for c in master.columns]
    partner.columns = [clean_text(c) for c in partner.columns]
    naics.columns = [clean_text(c) for c in naics.columns]

    for df in [master, partner, naics]:
        for col in df.columns:
            df[col] = df[col].apply(clean_text)

    if "Hiscox_COB" not in master.columns:
        raise ValueError("Hiscox543 sheet must contain a 'Hiscox_COB' column.")

    master["COB_NORM"] = master["Hiscox_COB"].apply(norm)
    valid_master_cobs = set(master["COB_NORM"].dropna().tolist())

    if "Active" in partner.columns:
        active_nonblank = partner["Active"].apply(norm).ne("").sum()
        if active_nonblank > 0:
            partner = partner[partner["Active"].apply(looks_like_active_yes)].copy()

    for col in ["Partner", "Partner_Description", "Partner_Secondary_Description", "Hiscox_COB"]:
        if col not in partner.columns:
            partner[col] = ""

    partner["PARTNER_NORM"] = partner["Partner"].apply(norm)
    partner["DESC_NORM"] = partner["Partner_Description"].apply(apply_synonyms)
    partner["SECONDARY_NORM"] = partner["Partner_Secondary_Description"].apply(apply_synonyms)
    partner["COB_NORM"] = partner["Hiscox_COB"].apply(norm)

    partner = partner[
        partner["COB_NORM"].isin(valid_master_cobs) | (partner["COB_NORM"] == "ooa")
    ].copy()

    for col in ["Hiscox_COB", "NAICS_Description", "NAICS_Title", "NAICS_Code"]:
        if col not in naics.columns:
            naics[col] = ""

    naics["NAICS_DESC_NORM"] = naics["NAICS_Description"].apply(apply_synonyms)
    naics["NAICS_TITLE_NORM"] = naics["NAICS_Title"].apply(apply_synonyms)
    naics["NAICS_CODE_NORM"] = naics["NAICS_Code"].apply(lambda x: re.sub(r"\D", "", str(x)))
    naics["COB_NORM"] = naics["Hiscox_COB"].apply(norm)

    naics = naics[
        naics["COB_NORM"].isin(valid_master_cobs) | (naics["COB_NORM"] == "ooa")
    ].copy()

    master_choices = master["Hiscox_COB"].dropna().astype(str).tolist()
    master_choices_sorted = sorted(set(master_choices), key=lambda x: x.lower())

    return master, partner, naics, master_choices, master_choices_sorted

master_df, partner_df, naics_df, master_choices, master_choices_sorted = load_data()

# =========================
# HINTS / PATCH V1
# =========================
def get_weighted_hints(q_clean):
    """
    Returns:
      - forced_candidates: list of very strong target suggestions
      - weighted_hints: dict of target_cob -> bonus
    """
    forced_candidates = []
    weighted_hints = {}

    def add_hint(cob_name, bonus):
        weighted_hints[cob_name] = weighted_hints.get(cob_name, 0) + bonus

    q = q_clean

    # -------------------------
    # Dealer / distributor / wholesale
    # -------------------------
    has_dealerish = any(t in q for t in [
        "dealer", "dealers", "distributor", "distributors",
        "wholesale", "wholesaler", "wholesalers"
    ])

    carve_out_retail = any(t in q for t in [
        "book", "poster", "art", "gallery", "store", "retail", "apparel", "clothing"
    ])

    carve_out_auto = "auto" in q or "car " in q or q.startswith("car") or "truck" in q
    carve_out_materials = "construction materials" in q or "masonry" in q or "hardwood" in q or "flooring" in q

    if has_dealerish:
        if not (carve_out_retail or carve_out_auto or carve_out_materials):
            add_hint("Distribution, Wholesalers, Dealers and Other Sales", 10)

    # -------------------------
    # Common-sense boosts
    # -------------------------
    if "ad agency" in q or ("advertising" in q and "agency" in q):
        add_hint("Advertising", 18)

    if "eye doctor" in q or "optomet" in q or "ophthalm" in q:
        add_hint("Optometrist/Ophthalmologist", 20)

    if "dog daycare" in q or ("pet" in q and "daycare" in q):
        add_hint("Pet daycare", 18)

    if "computer consultant" in q or "programmer" in q or "software developer" in q:
        add_hint("IT Consulting", 16)

    if "apparel retailer" in q or "clothing store" in q or "apparel store" in q:
        add_hint("Clothing/apparel stores", 18)

    if "air conditioning service" in q or "hvac" in q:
        add_hint("Air conditioning systems installation and repair", 20)

    if "builder" in q:
        add_hint("General Contractor - Construction and installation only", 14)

    if "deck builder" in q:
        add_hint("Carpentry (interior & exterior)", 20)

    if "electrical contractor" in q or ("electrical" in q and "contractor" in q):
        add_hint("Electrical work (interior only)", 20)

    if "roofing contractor" in q or ("roofing" in q and "contractor" in q):
        add_hint("Roofing", 18)

    if "auto dealer" in q or ("auto" in q and "sales" in q):
        add_hint("OOA", 20)

    if "auto repair" in q or ("mechanical" in q and "auto" in q):
        add_hint("OOA", 20)

    if "contract carrier" in q or "trucking" in q:
        add_hint("OOA", 16)

    if "assisted living" in q:
        add_hint("OOA", 18)

    if "adult day health" in q or "adhc" in q:
        add_hint("OOA", 18)

    if "animal hospital" in q:
        add_hint("OOA", 14)

    if "construction materials" in q:
        add_hint("OOA", 16)

    if "health club" in q:
        add_hint("OOA", 14)

    if "health food store" in q:
        add_hint("OOA", 14)

    if "full service restaurant" in q:
        add_hint("OOA", 10)

    if "book store" in q or "bookstore" in q:
        add_hint("Book & Magazine Stores", 20)

    if "poster store" in q or ("poster" in q and "store" in q):
        add_hint("Art, Picture, & Poster Stores", 20)

    if "art gallery" in q or ("gallery" in q and "art" in q):
        add_hint("Art Gallery/Dealer", 18)

    return forced_candidates, weighted_hints

# =========================
# OVERRIDE RULES
# =========================
def check_overrides(q):
    # Hard overrides only where we are comfortable being very direct
    if any(t in q for t in ["church", "temple", "mosque", "worship"]):
        return {
            "target_cob": "Religious",
            "logic": "Override rule",
            "confidence": 100,
            "matched_text": "religious override",
            "source_type": "override",
            "raw_score": None,
            "partner": ""
        }

    if ("doctor" in q or "doctors" in q) and not any(
        t in q for t in ["psych", "psychi", "veterin", "vet", "dental", "therapy", "clinic", "eye", "optomet", "ophthalm"]
    ):
        return {
            "target_cob": "Physicians Office",
            "logic": "Override rule",
            "confidence": 100,
            "matched_text": "doctor override",
            "source_type": "override",
            "raw_score": None,
            "partner": ""
        }

    return None

# =========================
# SCORING HELPERS
# =========================
def build_candidate(target_cob, logic, confidence, matched_text, source_type, raw_score=None, partner_name=None):
    return {
        "target_cob": target_cob,
        "logic": logic,
        "confidence": int(round(confidence)),
        "matched_text": matched_text,
        "source_type": source_type,
        "raw_score": None if raw_score is None else int(round(raw_score)),
        "partner": partner_name or ""
    }

def apply_hint_bonus(candidate, weighted_hints):
    bonus = weighted_hints.get(candidate["target_cob"], 0)
    if bonus:
        candidate["confidence"] = int(min(100, candidate["confidence"] + bonus))
        candidate["logic"] = f"{candidate['logic']} + hint"
    return candidate

def dedupe_best_candidates(candidates, limit=3):
    best = {}
    for c in candidates:
        key = norm(c["target_cob"]) + "||" + c["logic"] + "||" + c.get("partner", "")
        if key not in best or c["confidence"] > best[key]["confidence"]:
            best[key] = c

    ranked = sorted(
        best.values(),
        key=lambda x: (
            x["confidence"],
            -abs(len(norm(x["matched_text"])) - len(norm(x["target_cob"]))),
            1 if x["source_type"] == "override" else
            2 if x["source_type"] == "partner_exact" else
            3 if x["source_type"] == "naics_exact" else
            4 if x["source_type"] == "partner_fuzzy" else
            5 if x["source_type"] == "naics_fuzzy" else
            6
        ),
        reverse=True
    )
    return ranked[:limit]

# =========================
# MATCH BUCKETS
# =========================
def exact_partner_match(q_clean):
    candidates = []
    hits = partner_df[partner_df["DESC_NORM"] == q_clean]

    for _, r in hits.iterrows():
        candidates.append(build_candidate(
            target_cob=r["Hiscox_COB"],
            logic="Exact partner shortcut",
            confidence=98,
            matched_text=r["Partner_Description"],
            source_type="partner_exact",
            partner_name=r["Partner"]
        ))
    return candidates

def exact_naics_match(q_clean):
    candidates = []

    hits = naics_df[naics_df["NAICS_DESC_NORM"] == q_clean]
    for _, r in hits.iterrows():
        candidates.append(build_candidate(
            target_cob=r["Hiscox_COB"],
            logic="Exact NAICS description",
            confidence=97,
            matched_text=r["NAICS_Description"],
            source_type="naics_exact"
        ))

    hits = naics_df[naics_df["NAICS_TITLE_NORM"] == q_clean]
    for _, r in hits.iterrows():
        candidates.append(build_candidate(
            target_cob=r["Hiscox_COB"],
            logic="Exact NAICS title",
            confidence=95,
            matched_text=r["NAICS_Title"],
            source_type="naics_exact"
        ))

    q_digits = re.sub(r"\D", "", q_clean)
    if q_digits and len(q_digits) == 6:
        hits = naics_df[naics_df["NAICS_CODE_NORM"] == q_digits]
        for _, r in hits.iterrows():
            candidates.append(build_candidate(
                target_cob=r["Hiscox_COB"],
                logic="Exact NAICS code",
                confidence=99,
                matched_text=str(r["NAICS_Code"]),
                source_type="naics_exact"
            ))

    return candidates

def fuzzy_partner_candidates(q_clean, limit=8):
    phrases = partner_df["DESC_NORM"].dropna().tolist()
    phrases = [p for p in phrases if p]

    if not phrases:
        return []

    raw = process.extract(q_clean, list(set(phrases)), scorer=fuzz.token_set_ratio, limit=limit)
    candidates = []

    for phrase, score, _ in raw:
        if score < 82:
            continue

        matching_rows = partner_df[partner_df["DESC_NORM"] == phrase]
        for _, r in matching_rows.iterrows():
            length_penalty = abs(len(phrase) - len(q_clean)) * 0.35

            # Base fuzzy partner confidence
            confidence = score + 4 - length_penalty

            # Talage penalty: can suggest, not dominate
            partner_name = clean_text(r["Partner"])
            if norm(partner_name) == "talage":
                confidence -= 8

            # London / Insureon still slightly more trusted than fuzzy Talage
            if norm(partner_name) in {"london", "insureon"}:
                confidence += 1

            confidence = min(94, confidence)

            candidates.append(build_candidate(
                target_cob=r["Hiscox_COB"],
                logic="Fuzzy partner shortcut",
                confidence=confidence,
                matched_text=r["Partner_Description"],
                source_type="partner_fuzzy",
                raw_score=score,
                partner_name=partner_name
            ))

    return candidates

def fuzzy_naics_candidates(q_clean, limit=8):
    candidates = []

    desc_choices = naics_df["NAICS_DESC_NORM"].dropna().tolist()
    desc_choices = [p for p in desc_choices if p]

    title_choices = naics_df["NAICS_TITLE_NORM"].dropna().tolist()
    title_choices = [p for p in title_choices if p]

    if desc_choices:
        raw_desc = process.extract(q_clean, list(set(desc_choices)), scorer=fuzz.token_set_ratio, limit=limit)
        for phrase, score, _ in raw_desc:
            if score < 80:
                continue
            rows = naics_df[naics_df["NAICS_DESC_NORM"] == phrase]
            for _, r in rows.iterrows():
                length_penalty = abs(len(phrase) - len(q_clean)) * 0.30
                confidence = min(95, score + 4 - length_penalty)
                candidates.append(build_candidate(
                    target_cob=r["Hiscox_COB"],
                    logic="Fuzzy NAICS description",
                    confidence=confidence,
                    matched_text=r["NAICS_Description"],
                    source_type="naics_fuzzy",
                    raw_score=score
                ))

    if title_choices:
        raw_title = process.extract(q_clean, list(set(title_choices)), scorer=fuzz.token_set_ratio, limit=limit)
        for phrase, score, _ in raw_title:
            if score < 84:
                continue
            rows = naics_df[naics_df["NAICS_TITLE_NORM"] == phrase]
            for _, r in rows.iterrows():
                length_penalty = abs(len(phrase) - len(q_clean)) * 0.25
                confidence = min(93, score + 2 - length_penalty)
                candidates.append(build_candidate(
                    target_cob=r["Hiscox_COB"],
                    logic="Fuzzy NAICS title",
                    confidence=confidence,
                    matched_text=r["NAICS_Title"],
                    source_type="naics_fuzzy",
                    raw_score=score
                ))

    return candidates

def fuzzy_cob_fallback(q_clean):
    raw = process.extract(q_clean, master_choices, scorer=fuzz.token_set_ratio, limit=5)
    candidates = []

    for phrase, score, _ in raw:
        if score < 88:
            continue

        length_penalty = abs(len(norm(phrase)) - len(q_clean)) * 0.40
        confidence = min(89, score - 3 - length_penalty)

        candidates.append(build_candidate(
            target_cob=phrase,
            logic="Fuzzy COB fallback",
            confidence=confidence,
            matched_text=phrase,
            source_type="cob_fallback",
            raw_score=score
        ))

    return candidates

# =========================
# APPETITE / MASTER LOOKUP
# =========================
def get_appetite_data(cob_name):
    cob_norm = norm(cob_name)

    if cob_norm == "ooa":
        return {
            "real_name": "OOA",
            "status": "OUT OF APPETITE",
            "css": "app-no",
            "gl": False,
            "pl": False,
            "bop": False,
            "cyb": False,
            "defn": "Mapped to out-of-appetite in helper data.",
            "restr": "",
            "group": "OOA Industry",
            "industry_code": ""
        }

    hit = master_df[master_df["COB_NORM"] == cob_norm]
    if hit.empty:
        return None

    row = hit.iloc[0]

    gl = normalize_yes_no(row.get("GL", ""))
    pl = normalize_yes_no(row.get("PL", ""))
    bop = normalize_yes_no(row.get("BOP", ""))
    cyb = normalize_yes_no(row.get("Cyber", ""))

    yes_flags = [name for name, flag in [("GL", gl), ("PL", pl), ("BOP", bop), ("Cyber", cyb)] if flag]

    if len(yes_flags) >= 2:
        status = "IN APPETITE"
        css = "app-yes"
    elif len(yes_flags) == 1:
        status = f"LIMITED APPETITE ({yes_flags[0]} ONLY)"
        css = "app-yes"
    else:
        status = "OUT OF APPETITE"
        css = "app-no"

    return {
        "real_name": row.get("Hiscox_COB", cob_name),
        "status": status,
        "css": css,
        "gl": gl,
        "pl": pl,
        "bop": bop,
        "cyb": cyb,
        "defn": row.get("Definition", ""),
        "restr": row.get("State Restrictions", ""),
        "group": row.get("COB_Group", ""),
        "industry_code": row.get("Full_Industry_Code", "")
    }

# =========================
# CORE SEARCH ENGINE
# =========================
def search_engine(raw_query):
    q_clean = apply_synonyms(raw_query)
    all_candidates = []

    # Hard overrides
    override = check_overrides(q_clean)
    if override:
        all_candidates.append(override)

    # Weighted hints
    forced_candidates, weighted_hints = get_weighted_hints(q_clean)
    for forced in forced_candidates:
        all_candidates.append(forced)

    # Exact layers
    exact_partner = exact_partner_match(q_clean)
    exact_naics = exact_naics_match(q_clean)

    exact_partner = [apply_hint_bonus(c, weighted_hints) for c in exact_partner]
    exact_naics = [apply_hint_bonus(c, weighted_hints) for c in exact_naics]

    all_candidates.extend(exact_partner)
    all_candidates.extend(exact_naics)

    exactish = [c for c in all_candidates if c["source_type"] in {"override", "partner_exact", "naics_exact"}]
    if exactish:
        ranked = dedupe_best_candidates(exactish, limit=3)
        return ranked[0], ranked, q_clean

    # Fuzzy layers
    partner_fuzzy = fuzzy_partner_candidates(q_clean, limit=10)
    naics_fuzzy = fuzzy_naics_candidates(q_clean, limit=10)

    partner_fuzzy = [apply_hint_bonus(c, weighted_hints) for c in partner_fuzzy]
    naics_fuzzy = [apply_hint_bonus(c, weighted_hints) for c in naics_fuzzy]

    all_candidates.extend(partner_fuzzy)
    all_candidates.extend(naics_fuzzy)

    if not all_candidates:
        cob_fallback = fuzzy_cob_fallback(q_clean)
        cob_fallback = [apply_hint_bonus(c, weighted_hints) for c in cob_fallback]
        all_candidates.extend(cob_fallback)

    ranked = dedupe_best_candidates(all_candidates, limit=3)

    if not ranked:
        return None, [], q_clean

    best = ranked[0]

    accepted = False
    if best["source_type"] == "partner_fuzzy" and best["confidence"] >= 86:
        accepted = True
    elif best["source_type"] == "naics_fuzzy" and best["confidence"] >= 82:
        accepted = True
    elif best["source_type"] == "cob_fallback" and best["confidence"] >= 90:
        accepted = True

    if accepted:
        return best, ranked, q_clean

    return None, ranked, q_clean

# =========================
# REVIEW RECORD HELPERS
# =========================
def candidate_field(candidates, idx, field):
    if len(candidates) > idx:
        return candidates[idx].get(field, "")
    return ""

def build_batch_review_record(raw_val, best_match, top_candidates, q_clean):
    appetite = get_appetite_data(best_match["target_cob"]) if best_match else None
    engine_choice = appetite["real_name"] if appetite else (best_match["target_cob"] if best_match else "")

    return {
        "Input_Description": raw_val,
        "Engine_Chosen_COB": engine_choice,
        "Candidate_1": candidate_field(top_candidates, 0, "target_cob"),
        "Candidate_2": candidate_field(top_candidates, 1, "target_cob"),
        "Candidate_3": candidate_field(top_candidates, 2, "target_cob"),
        "Final_Selected_COB": engine_choice,
        "Reviewer_Notes": "",

        # Hidden-ish metadata retained for export
        "Cleaned_Query": q_clean,
        "Engine_Logic": best_match["logic"] if best_match else "No confident match",
        "Engine_Confidence": best_match["confidence"] if best_match else "",
        "Engine_Partner_Source": best_match.get("partner", "") if best_match else "",
        "Candidate_1_Logic": candidate_field(top_candidates, 0, "logic"),
        "Candidate_1_Confidence": candidate_field(top_candidates, 0, "confidence"),
        "Candidate_1_Partner": candidate_field(top_candidates, 0, "partner"),
        "Candidate_2_Logic": candidate_field(top_candidates, 1, "logic"),
        "Candidate_2_Confidence": candidate_field(top_candidates, 1, "confidence"),
        "Candidate_2_Partner": candidate_field(top_candidates, 1, "partner"),
        "Candidate_3_Logic": candidate_field(top_candidates, 2, "logic"),
        "Candidate_3_Confidence": candidate_field(top_candidates, 2, "confidence"),
        "Candidate_3_Partner": candidate_field(top_candidates, 2, "partner"),
    }

def build_batch_results_export(review_rows):
    df = pd.DataFrame(review_rows)

    # Put user-facing columns first
    preferred = [
        "Input_Description",
        "Engine_Chosen_COB",
        "Candidate_1",
        "Candidate_2",
        "Candidate_3",
        "Final_Selected_COB",
        "Reviewer_Notes",
        "Cleaned_Query",
        "Engine_Logic",
        "Engine_Confidence",
        "Engine_Partner_Source",
        "Candidate_1_Logic",
        "Candidate_1_Confidence",
        "Candidate_1_Partner",
        "Candidate_2_Logic",
        "Candidate_2_Confidence",
        "Candidate_2_Partner",
        "Candidate_3_Logic",
        "Candidate_3_Confidence",
        "Candidate_3_Partner",
    ]
    cols = [c for c in preferred if c in df.columns]
    df = df[cols]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Reviewed_Batch")
        ref_df = pd.DataFrame({"Valid_Hiscox_COB": master_choices_sorted})
        ref_df.to_excel(writer, index=False, sheet_name="Valid_Hiscox_COBs")

        ws = writer.book["Reviewed_Batch"]
        ws.freeze_panes = "A2"

        for column_cells in ws.columns:
            max_length = 0
            col_letter = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max(max_length + 2, 14), 40)

    output.seek(0)
    return output.getvalue()

# =========================
# RENDER
# =========================
def display_verdict(cob_name):
    data = get_appetite_data(cob_name)

    if not data:
        st.error(f"⚠️ Mapped to '{cob_name}', but it is missing from Hiscox543.")
        return

    st.markdown(
        f'<div class="scorecard {data["css"]}"><h2>{data["status"]}</h2><p>Hiscox Class: <b>{data["real_name"]}</b></p></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    for label, flag, col in [
        ("GL", data["gl"], c1),
        ("PL", data["pl"], c2),
        ("BOP", data["bop"], c3),
        ("Cyber", data["cyb"], c4),
    ]:
        css = "lob-yes" if flag else "lob-no"
        text = "YES" if flag else "NO"
        col.markdown(f'<div class="lob-box {css}">{label}<br>{text}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info(f"**📖 Definition:** {data['defn'] if data['defn'] else 'No specific definition provided.'}")

    if data["restr"]:
        st.warning(f"**🚧 Restrictions:** {data['restr']}")

def render_diagnostics(raw_query, q_clean, best_match, top_candidates):
    with st.expander("⚙️ Tuning & Diagnostics", expanded=True):
        st.write(f"**Original query:** `{raw_query}`")
        st.write(f"**Cleaned query:** `{q_clean}`")

        if best_match:
            st.write("---")
            st.write("**Chosen match**")
            st.write(f"- Logic: {best_match['logic']}")
            st.write(f"- Target COB: {best_match['target_cob']}")
            st.write(f"- Confidence: {best_match['confidence']}%")
            if best_match.get("raw_score") is not None:
                st.write(f"- Raw fuzzy score: {best_match['raw_score']}%")
            st.write(f"- Matched text: `{best_match['matched_text']}`")
            if best_match.get("partner"):
                st.write(f"- Partner shortcut source: **{best_match['partner']}**")

        st.write("---")
        st.write("**Top engine candidates**")
        if top_candidates:
            for i, c in enumerate(top_candidates, start=1):
                line = f"{i}. **{c['target_cob']}** — {c['logic']} — {c['confidence']}%"
                if c.get("raw_score") is not None:
                    line += f" (raw {c['raw_score']}%)"
                st.write(line)
                st.caption(f"Matched text: {c['matched_text']}")
                if c.get("partner"):
                    st.caption(f"Partner source: {c['partner']}")
        else:
            st.write("No candidates found.")

# =========================
# SINGLE REVIEW UI
# =========================
def render_feedback_ui(raw_query, q_clean, best_match, top_candidates):
    st.markdown("---")
    st.subheader("📝 Review This Match")

    candidate_1 = top_candidates[0] if len(top_candidates) > 0 else None
    candidate_2 = top_candidates[1] if len(top_candidates) > 1 else None
    candidate_3 = top_candidates[2] if len(top_candidates) > 2 else None

    widget_key_base = re.sub(r"[^a-zA-Z0-9_]", "_", q_clean)[:80] or "query"

    options = []
    option_map = {}

    if candidate_1:
        label = f"Accept current match — {candidate_1['target_cob']}"
        options.append(label)
        option_map[label] = ("candidate_1", candidate_1["target_cob"])

    if candidate_2:
        label = f"Use candidate 2 — {candidate_2['target_cob']}"
        options.append(label)
        option_map[label] = ("candidate_2", candidate_2["target_cob"])

    if candidate_3:
        label = f"Use candidate 3 — {candidate_3['target_cob']}"
        options.append(label)
        option_map[label] = ("candidate_3", candidate_3["target_cob"])

    none_label = "None of these / choose a different COB"
    options.append(none_label)
    option_map[none_label] = ("manual", "")

    selected_action = st.radio(
        "How should this result be logged?",
        options=options,
        index=0,
        key=f"feedback_choice_{widget_key_base}"
    )

    manual_cob = ""
    if option_map[selected_action][0] == "manual":
        manual_cob = st.selectbox(
            "Choose the correct Hiscox_COB",
            options=[""] + master_choices_sorted,
            index=0,
            key=f"manual_cob_{widget_key_base}"
        )

    notes = st.text_input(
        "Optional notes",
        placeholder="Why was this changed?",
        key=f"feedback_notes_{widget_key_base}"
    )

    if st.button("Save Feedback", key=f"save_feedback_{widget_key_base}"):
        selection_type, selected_cob = option_map[selected_action]

        if selection_type == "manual":
            if not manual_cob:
                st.warning("Please choose a correct Hiscox_COB before saving.")
                return
            final_selection = manual_cob
        else:
            final_selection = selected_cob

        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input_Query": raw_query,
            "Cleaned_Query": q_clean,
            "Engine_Chosen_COB": best_match["target_cob"] if best_match else "",
            "Engine_Logic": best_match["logic"] if best_match else "No confident match",
            "Engine_Confidence": best_match["confidence"] if best_match else "",
            "Engine_Partner_Source": best_match.get("partner", "") if best_match else "",
            "Candidate_1": candidate_1["target_cob"] if candidate_1 else "",
            "Candidate_1_Logic": candidate_1["logic"] if candidate_1 else "",
            "Candidate_1_Confidence": candidate_1["confidence"] if candidate_1 else "",
            "Candidate_1_Partner": candidate_1.get("partner", "") if candidate_1 else "",
            "Candidate_2": candidate_2["target_cob"] if candidate_2 else "",
            "Candidate_2_Logic": candidate_2["logic"] if candidate_2 else "",
            "Candidate_2_Confidence": candidate_2["confidence"] if candidate_2 else "",
            "Candidate_2_Partner": candidate_2.get("partner", "") if candidate_2 else "",
            "Candidate_3": candidate_3["target_cob"] if candidate_3 else "",
            "Candidate_3_Logic": candidate_3["logic"] if candidate_3 else "",
            "Candidate_3_Confidence": candidate_3["confidence"] if candidate_3 else "",
            "Candidate_3_Partner": candidate_3.get("partner", "") if candidate_3 else "",
            "User_Selection_Type": selection_type,
            "Final_Selected_COB": final_selection,
            "Notes": notes
        }

        st.session_state.feedback_log.append(row)
        st.success("Feedback saved to Correction Log.")

def render_feedback_log():
    st.markdown("---")
    st.header("📋 Correction Log")

    if not st.session_state.feedback_log:
        st.write("No feedback saved yet.")
        return

    feedback_df = pd.DataFrame(st.session_state.feedback_log)
    st.dataframe(feedback_df, use_container_width=True)

    csv_buffer = io.StringIO()
    feedback_df.to_csv(csv_buffer, index=False)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download Correction Log (.csv)",
            data=csv_buffer.getvalue(),
            file_name="Atlas_Correction_Log.csv",
            mime="text/csv"
        )

    with col2:
        if st.button("Clear Feedback Log"):
            st.session_state.feedback_log = []
            st.experimental_rerun()

# =========================
# MAIN UI
# =========================
st.title("🛡️ Atlas 2026: Underwriting Engine")
st.caption("Prototype using Hiscox543 + Partner_Mapping + NAICS_Mapping")

query = st.text_input(
    "Search for a business description or 6-digit NAICS code",
    placeholder="e.g., PR firm, jeweler, yoga studio, 541820",
    label_visibility="collapsed"
)

if query:
    best_match, top_candidates, q_clean = search_engine(query)

    if best_match:
        display_verdict(best_match["target_cob"])
    else:
        st.error("❌ No match found with enough confidence. See diagnostics below.")

    render_diagnostics(
        raw_query=query,
        q_clean=q_clean,
        best_match=best_match,
        top_candidates=top_candidates
    )

    render_feedback_ui(
        raw_query=query,
        q_clean=q_clean,
        best_match=best_match,
        top_candidates=top_candidates
    )

render_feedback_log()

# =========================
# BATCH REVIEW UI V2
# =========================
st.markdown("---")
st.header("📥 Batch Review")
st.write("Upload a CSV or Excel file, review matches in the app, then download the reviewed results.")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="batch_uploader_v2")

if uploaded_file:
    if st.session_state.batch_review_filename != uploaded_file.name:
        with st.spinner("Preparing batch review..."):
            if uploaded_file.name.lower().endswith(".csv"):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)

            candidate_cols = []
            for col in upload_df.columns:
                try:
                    series = upload_df[col].fillna("").astype(str)
                    avg_len = series.str.len().mean()
                    nonblank = (series.str.strip() != "").mean()
                    header_score = 0
                    col_norm = norm(col)

                    if any(k in col_norm for k in ["description", "industry", "business", "naics", "class", "activity", "partner"]):
                        header_score += 20
                    if avg_len > 6:
                        header_score += 10
                    if nonblank > 0.4:
                        header_score += 10

                    candidate_cols.append((col, header_score))
                except Exception:
                    pass

            candidate_cols = sorted(candidate_cols, key=lambda x: x[1], reverse=True)
            text_column = candidate_cols[0][0] if candidate_cols else None

            if not text_column:
                st.error("Could not auto-detect a description column.")
            else:
                prepared_rows = []
                for raw_val in upload_df[text_column].fillna("").astype(str):
                    if not raw_val.strip():
                        continue
                    best_match, top_candidates, q_clean = search_engine(raw_val)
                    prepared_rows.append(build_batch_review_record(raw_val, best_match, top_candidates, q_clean))

                st.session_state.batch_review_rows = prepared_rows
                st.session_state.batch_review_filename = uploaded_file.name

if st.session_state.batch_review_rows:
    st.success(f"Loaded batch review set from: {st.session_state.batch_review_filename}")

    st.markdown('<div class="small-note">Edit only the rows that need correction. The final selected COB is prefilled with the engine choice when one exists.</div>', unsafe_allow_html=True)

    for i, row in enumerate(st.session_state.batch_review_rows):
        with st.expander(f"{i+1}. {row['Input_Description']}", expanded=(i < 5)):
            st.write(f"**Engine choice:** {row['Engine_Chosen_COB'] or 'No confident match'}")

            candidate_text = []
            for n in [1, 2, 3]:
                cand = row.get(f"Candidate_{n}", "")
                if cand:
                    partner_src = row.get(f"Candidate_{n}_Partner", "")
                    logic = row.get(f"Candidate_{n}_Logic", "")
                    conf = row.get(f"Candidate_{n}_Confidence", "")
                    extra = f" — {logic} — {conf}%"
                    if partner_src:
                        extra += f" — {partner_src}"
                    candidate_text.append(f"{n}. {cand}{extra}")

            if candidate_text:
                st.write("**Top candidates:**")
                for line in candidate_text:
                    st.write(f"- {line}")
            else:
                st.write("**Top candidates:** None")

            current_default = row.get("Final_Selected_COB", "")
            try:
                default_index = ([""] + master_choices_sorted).index(current_default) if current_default else 0
            except ValueError:
                default_index = 0

            selected_cob = st.selectbox(
                "Final selected Hiscox_COB",
                options=[""] + master_choices_sorted,
                index=default_index,
                key=f"batch_final_cob_{i}"
            )
            row["Final_Selected_COB"] = selected_cob

            notes_val = st.text_input(
                "Reviewer notes",
                value=row.get("Reviewer_Notes", ""),
                key=f"batch_notes_{i}"
            )
            row["Reviewer_Notes"] = notes_val

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        reviewed_excel = build_batch_results_export(st.session_state.batch_review_rows)
        st.download_button(
            label="⬇️ Download Reviewed Batch (.xlsx)",
            data=reviewed_excel,
            file_name="Atlas_Batch_Reviewed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        if st.button("Clear Batch Review Set"):
            st.session_state.batch_review_rows = []
            st.session_state.batch_review_filename = ""
            st.experimental_rerun()
