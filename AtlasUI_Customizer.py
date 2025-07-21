
import streamlit as st

# === PAGE CONFIG ===
st.set_page_config(layout="centered", page_title="Atlas UI Customizer")

# === CUSTOM STYLING ===
st.markdown("""
<style>
body, .stApp, .block-container {
    background-color: #0f0f0f;
    color: #f1f1f1;
}
.stTextInput input {
    background-color: #1f1f1f;
    color: #ffffff;
    border-radius: 8px;
}
.stButton>button, .stDownloadButton>button {
    background-color: #e63946;
    color: white;
    border-radius: 12px;
    font-weight: bold;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #a4161a;
}
hr {
    border-top: 2px solid #e63946;
    margin-top: 1em;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h1 style='text-align:center;'>🧪 Atlas UI Sandbox</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>This version is for testing themes, layout, fonts, and styling only.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# === PREVIEW ELEMENTS ===
st.markdown("### 🔍 Search Bar Example")
st.text_input("Enter something...")

st.markdown("### 📥 Upload Button Example")
st.file_uploader("Upload something...")

st.markdown("### 📊 Data Table Preview")
st.dataframe({"Example": ["Row 1", "Row 2"], "Value": [123, 456]})

st.markdown("### 🎛️ Buttons")
st.button("Run Mock Match")
st.download_button("⬇️ Download Placeholder", data="Test", file_name="mock.csv")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Use this to fine-tune the user-facing interface before embedding into full logic.</p>", unsafe_allow_html=True)
