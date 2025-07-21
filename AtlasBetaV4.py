# === IMPORTS ===
import pandas as pd
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image

st.set_page_config(layout="centered")

# === Custom Black/Red Theme + Drag Highlight ===
st.markdown("""
    <style>
        body, .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stTextInput > div > div > input {
            background-color: #222222;
            color: #ffffff;
        }
        .stFileUploader, .stButton {
            background-color: #111111;
        }
        .drop-target-active {
            border: 2px dashed #ff4b4b !important;
            background-color: #1a1a1a !important;
            border-radius: 8px !important;
        }
        .appetite-button {
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: bold;
        }
        .green-btn { background-color: #00cc66; color: black; }
        .red-btn { background-color: #cc3333; color: white; }
    </style>
    <script>
        const observer = new MutationObserver(() => {
            const dropzone = document.querySelector('section[data-testid="stFileUploader"]');
            if (dropzone && !dropzone.classList.contains("watched")) {
                dropzone.classList.add("watched");
                dropzone.addEventListener("dragenter", function () {
                    dropzone.classList.add("drop-target-active");
                });
                dropzone.addEventListener("dragleave", function () {
                    dropzone.classList.remove("drop-target-active");
                });
                dropzone.addEventListener("drop", function () {
                    dropzone.classList.remove("drop-target-active");
                });
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)

# === Load Logo ===
logo = Image.open("AtlasLogo.jpeg")
st.image(logo, use_container_width=False, width=300)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_excel("AtlasEngine.xlsx", sheet_name="NAICS_COB")
    for col in ["Hiscox_COB", "NAICS_Title", "NAICS_Description", "COB_Group"]:
        df[col] = df[col].astype(str).str.strip()
    df["match_corpus"] = (
        df["NAICS_Description"] + " | " +
        df["NAICS_Title"] + " | " +
        df["COB_Group"] + " | " +
        df["Hiscox_COB"]
    )
    return df

engine_df = load_data()

@st.cache_resource
def build_model_index(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus.tolist(), convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

model, index = build_model_index(engine_df["match_corpus"])

# === Matching Logic ===
def summarize_appetite_logic(row):
    flags = [row["PL"], row["GL"], row["BOP"], row["Cyber"]]
    labels = ["PL", "GL", "BOP", "Cyber"]
    yes_flags = [label for flag, label in zip(flags, labels) if flag == "Y"]
    if len(yes_flags) >= 2:
        return "In Appetite", "green-btn"
    elif len(yes_flags) == 1:
        return f"{yes_flags[0]} Only", "green-btn"
    else:
        return "Out of Appetite", "red-btn"

def confidence_label(score_pct):
    if score_pct >= 86:
        return f"{score_pct:.1f}% (High Confidence)"
    elif score_pct >= 70:
        return f"{score_pct:.1f}% (Needs Review)"
    else:
        return f"{score_pct:.1f}% (Low Confidence)"

def match_status(score_pct):
    if score_pct >= 86:
        return "Confirmed"
    elif score_pct >= 70:
        return "Needs Review"
    else:
        return "No Match Found"
# === OVERRIDE RULES BLOCK FOR STRONG KEYWORDS ===
def check_override_rules(input_text, engine_df):
    text = input_text.lower().strip()

    # === Rule 1: Distributors / Dealers → Distribution COB ===
    if any(term in text for term in ["distributor", "distributors", "dealer"]):
        dist_row = engine_df[engine_df["Hiscox_COB"].str.contains("Distribution, Wholesalers, Dealers and Other Sales", case=False)]
        if not dist_row.empty:
            return dist_row.iloc[0], "Override: Distributor/Dealer keyword"

    # === Rule 2: Auto + Repair → Auto COB ===
    if "auto" in text and "repair" in text:
        auto_row = engine_df[engine_df["Hiscox_COB"].str.contains("Auto, Car, Truck, Boat", case=False)]
        if not auto_row.empty:
            return auto_row.iloc[0], "Override: Auto Repair keyword"

    # === Rule 3: Doctor(s) → Physicians Office ===
    if "doctor" in text or "doctors" in text:
        if not any(term in text for term in ["psych", "veterin", "dental", "therapy", "clinic"]):
            doc_row = engine_df[engine_df["Hiscox_COB"].str.lower() == "physicians office"]
            if not doc_row.empty:
                return doc_row.iloc[0], "Override: General Doctor(s)"

    # === Rule 4: Book alone ≠ Publishing unless NAICS confirms ===
    if "book" in text:
        if not any(term in text for term in ["publish", "print", "author", "editor", "ebook", "magazine", "journal", "media"]):
            blocklist = ["publishing", "printing"]
            match_candidates = engine_df[engine_df["Hiscox_COB"].str.lower().str.contains("|".join(blocklist))]
            if not match_candidates.empty:
                return None, "Override: Book keyword excluded from publishing/printing"

    return None, None

# === ORIGINAL FUNCTION CONTINUES ===
def check_tier1_rules(input_clean, engine_df):
    cob_match = engine_df[engine_df["Hiscox_COB"].str.lower() == input_clean]
    if not cob_match.empty:
        return cob_match.iloc[0], "Tier 1 Match: Exact Hiscox COB"
    naics_match = engine_df[engine_df["NAICS_Description"].str.lower() == input_clean]
    if not naics_match.empty:
        return naics_match.iloc[0], "Tier 1 Match: Exact NAICS Description"
    return None, None

# === Top 3 Search Logic ===
def search_top_matches(input_text):
    input_clean = str(input_text).strip().lower()
    # === Apply strong override rules ===
    override_row, override_reason = check_override_rules(input_text, engine_df)
    if override_row is not None:
        return [{
            "Input_Description": input_text,
            "Hiscox_COB": override_row["Hiscox_COB"],
            "full_industry_code": override_row.get("full_industry_code", ""),
            "Confidence (%)": "100.0% (Override Rule)",
            "Match_Status": "Confirmed via Override",
            "Appetite": summarize_appetite_logic(override_row),
            "PL": override_row["PL"],
            "GL": override_row["GL"],
            "BOP": override_row["BOP"],
            "Cyber": override_row["Cyber"]
        }]

    # NAICS Direct Search
    if input_clean.isdigit() and len(input_clean) == 6:
        matches = engine_df[engine_df["NAICS_Code"].astype(str).str.startswith(input_clean)]
        results = []
        for _, row in matches.iterrows():
            appetite_text, appetite_class = summarize_appetite_logic(row)
            results.append({
                "Input_Description": input_text,
                "Hiscox_COB": row["Hiscox_COB"],
                "Confidence": "N/A (NAICS Search)",
                "Match_Status": "Confirmed via NAICS",
                "Appetite": appetite_text,
                "AppetiteClass": appetite_class,
                "LOB_Details": row[["PL", "GL", "BOP", "Cyber"]].to_dict()
            })
        return results[:3]

    # Tier 1 Match
    tier1_row, _ = check_tier1_rules(input_clean, engine_df)
    if tier1_row is not None:
        appetite_text, appetite_class = summarize_appetite_logic(tier1_row)
        return [{
            "Input_Description": input_text,
            "Hiscox_COB": tier1_row["Hiscox_COB"],
            "Confidence": "100.0% (High Confidence)",
            "Match_Status": "Confirmed",
            "Appetite": appetite_text,
            "AppetiteClass": appetite_class,
            "LOB_Details": tier1_row[["PL", "GL", "BOP", "Cyber"]].to_dict()
        }]

    # Semantic Match
    embedding = model.encode([input_text], convert_to_numpy=True)
    D, I = index.search(embedding, 3)
    results = []
    for rank in range(3):
        idx = I[0][rank]
        row = engine_df.iloc[idx]
        score = 1 / (1 + D[0][rank])
        score_pct = round(score * 100, 1)
        appetite_text, appetite_class = summarize_appetite_logic(row)
        results.append({
            "Input_Description": input_text,
            "Hiscox_COB": row["Hiscox_COB"],
            "Confidence": confidence_label(score_pct),
            "Match_Status": match_status(score_pct),
            "Appetite": appetite_text,
            "AppetiteClass": appetite_class,
            "LOB_Details": row[["PL", "GL", "BOP", "Cyber"]].to_dict()
        })
    return results

# === Batch Matching ===
def run_batch_match(inputs):
    results = []
    progress_bar = st.progress(0)  # Initialize the progress bar
    total = len(inputs)

    embeddings = model.encode(inputs, convert_to_numpy=True)
    D, I = index.search(embeddings, 1)

    for i, entry in enumerate(inputs):
        input_clean = str(entry).strip().lower()
        tier1_row, _ = check_tier1_rules(input_clean, engine_df)

        if tier1_row is not None:
            appetite_text, _ = summarize_appetite_logic(tier1_row)
            results.append({
                "Input_Description": entry,
                "Hiscox_COB": tier1_row["Hiscox_COB"],
                "Confidence (%)": "100.0% (High Confidence)",
                "Match_Status": "Confirmed",
                "Appetite": appetite_text,
                "PL": tier1_row["PL"],
                "GL": tier1_row["GL"],
                "BOP": tier1_row["BOP"],
                "Cyber": tier1_row["Cyber"]
            })
        else:
            idx = I[i][0]
            row = engine_df.iloc[idx]
            sim_score = 1 / (1 + D[i][0])
            sim_pct = round(sim_score * 100, 1)
            appetite_text, _ = summarize_appetite_logic(row)
            results.append({
                "Input_Description": entry,
                "Hiscox_COB": row["Hiscox_COB"],
                "Confidence (%)": confidence_label(sim_pct),
                "Match_Status": match_status(sim_pct),
                "Appetite": appetite_text,
                "PL": row["PL"],
                "GL": row["GL"],
                "BOP": row["BOP"],
                "Cyber": row["Cyber"]
            })

        # Update progress bar
        progress_bar.progress((i + 1) / total)

    progress_bar.empty()  # Clear the bar when done
    return pd.DataFrame(results)


# === UI ===
st.text_input("🔍 Search for a business description", key="search_input")
search_input = st.session_state.get("search_input", "")
if search_input:
    st.markdown("#### Top 3 Matches:")
    results = search_top_matches(search_input)
    for res in results:
st.markdown(f"**{res['Hiscox_COB']}** — {res['Confidence (%)']} — *{res['Match_Status']}*")
        st.markdown(f"""
            <div class="appetite-button {res['AppetiteClass']}" title="
                PL: {'✅' if res['LOB_Details']['PL'] == 'Y' else '❌'} |
                GL: {'✅' if res['LOB_Details']['GL'] == 'Y' else '❌'} |
                BOP: {'✅' if res['LOB_Details']['BOP'] == 'Y' else '❌'} |
                Cyber: {'✅' if res['LOB_Details']['Cyber'] == 'Y' else '❌'}
            ">{res['Appetite']}</div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📥 Batch Class of Business Mapping")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    text_column = None
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 5:
            text_column = col
            break

    if text_column:
        result_df = run_batch_match(df[text_column].fillna(""))
        st.download_button(
            "⬇️ Download Match Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="Atlas_Match_Results.csv",
            mime="text/csv"
        )
    else:
        st.error("No suitable text column found in uploaded file.")
