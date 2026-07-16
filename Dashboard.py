"""
Patient Case Dashboard — lightweight Streamlit backup of the full webapp's
case workflow (dashboard / new case / case detail), no login required.
"""

import streamlit as st

from db import delete_all_cases, init_db, list_cases
from pipeline import load_attention_unet_ensemble, load_unet_ensemble, load_yolo

st.set_page_config(page_title="Lichen Detection — Case Dashboard", layout="wide")
init_db()

if "models_loaded" not in st.session_state:
    with st.spinner("Loading AI models (YOLO gate + UNet + Attention UNet ensembles)…"):
        load_yolo()
        load_unet_ensemble()
        load_attention_unet_ensemble()
    st.session_state["models_loaded"] = True

STATUS_LABEL = {"pending": "Pending", "analyzed": "Analyzed", "reviewed": "Reviewed", "finalized": "Finalized"}
CONCLUSION_LABEL = {
    "lichen_planus": "🔴 Lichen Planus",
    "other_lesion": "🟠 Other Lesion",
    "normal": "⬜ Normal Mucosa",
    "inconclusive": "Inconclusive",
}

st.title("Patient Case Dashboard")
st.caption("Case history and AI detection reports — lightweight backup instance")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("+ New Case", type="primary", use_container_width=True):
        st.switch_page("pages/1_New_Case.py")

cases = list_cases()

total = len(cases)
pending = sum(1 for c in cases if c["status"] == "pending")
lichen_positive = sum(1 for c in cases if c["overall_conclusion"] == "lichen_planus")

s1, s2, s3 = st.columns(3)
s1.metric("Total cases", total)
s2.metric("Pending analysis", pending)
s3.metric("Lichen planus positive", lichen_positive)

st.divider()

if not cases:
    st.info("No cases yet. Start with **+ New Case** above.")
else:
    header = st.columns([2, 2, 2, 1, 2, 2])
    for col, label in zip(header, ["Case", "Patient", "Date", "Images", "Status", "AI Conclusion"]):
        col.markdown(f"**{label}**")

    for case in cases:
        row = st.columns([2, 2, 2, 1, 2, 2])
        if row[0].button(case["case_code"], key=f"open_{case['id']}"):
            st.session_state["current_case_id"] = case["id"]
            st.switch_page("pages/2_Case_Detail.py")
        row[1].write(case["patient_code"])
        row[2].write(case["created_at"][:10])
        row[3].write(case["image_count"])
        row[4].write(STATUS_LABEL.get(case["status"], case["status"]))
        row[5].write(CONCLUSION_LABEL.get(case["overall_conclusion"], "—"))

st.divider()

if cases:
    with st.expander("⚠️ Danger zone"):
        st.warning("This permanently deletes every case, image, and note. This cannot be undone.")
        confirm = st.checkbox("I understand this will permanently delete all case data")
        if st.button("Remove All Cases", disabled=not confirm):
            delete_all_cases()
            st.success("All cases removed.")
            st.rerun()

st.divider()
st.caption(
    "⚠️ This AI tool is a diagnostic assistant and does not substitute for a professional medical diagnosis."
)
