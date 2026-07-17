"""
Patient Case Dashboard — lightweight Streamlit backup of the full webapp's
case workflow (dashboard / new case / case detail), no login required.
"""

import streamlit as st

from db import delete_all_cases, init_db, list_cases
from pipeline import load_unet_ensemble, load_yolo

st.set_page_config(page_title="Lichen Detection — Case Dashboard", layout="wide")
init_db()

if "models_loaded" not in st.session_state:
    with st.spinner("Loading AI models (YOLO gate + UNet ensemble)…"):
        load_yolo()
        load_unet_ensemble()
    st.session_state["models_loaded"] = True

CONCLUSION_LABEL = {
    "lichen_planus": "🔴 Lichen Planus",
    "other_lesion": "🟢 Other Lesion",
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

st.metric("Total cases", len(cases))

st.divider()

if not cases:
    st.info("No cases yet. Start with **+ New Case** above.")
else:
    header = st.columns([2, 2, 1, 2])
    for col, label in zip(header, ["Patient ID", "Date", "Number of Images", "Suggested Diagnosis"]):
        col.markdown(f"**{label}**")

    for case in cases:
        row = st.columns([2, 2, 1, 2])
        if row[0].button(case["patient_code"], key=f"open_{case['id']}"):
            st.session_state["current_case_id"] = case["id"]
            st.switch_page("pages/2_Case_Detail.py")
        row[1].write(case["created_at"][:10])
        row[2].write(case["image_count"])
        row[3].write(CONCLUSION_LABEL.get(case["overall_conclusion"], "—"))

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
