import streamlit as st

from db import CONCLUSION_COLOR, add_note, conclusion_reason, get_case

st.set_page_config(page_title="Case Detail — Lichen Detection", layout="wide")

STATUS_LABEL = {"pending": "Pending", "analyzed": "Analyzed", "reviewed": "Reviewed", "finalized": "Finalized"}
CONCLUSION_LABEL = {
    "lichen_planus": "Lichen Planus",
    "other_lesion": "Other Lesion",
    "normal": "Normal Mucosa",
    "inconclusive": "Inconclusive",
}
LABEL_DISPLAY = {
    "lichen": ("Lichen Planus", "🔴"),
    "other_lesion": ("Other Lesion", "🟢"),
    "normal": ("Normal Mucosa", "⬜"),
}

case_id = st.session_state.get("current_case_id") or st.query_params.get("case_id")

if st.button("← Back to dashboard"):
    st.session_state.pop("current_case_id", None)
    st.switch_page("Dashboard.py")

if not case_id:
    st.warning("No case selected.")
    st.stop()

case = get_case(case_id)
if not case:
    st.error("Case not found.")
    st.stop()

st.title(case["case_code"])
st.caption(f"Patient {case['patient_code']} · {case['created_at'][:10]}")

badge_col1, badge_col2 = st.columns([1, 1])
badge_col1.markdown(f"**Status:** {STATUS_LABEL.get(case['status'], case['status'])}")
if case["overall_conclusion"]:
    color = CONCLUSION_COLOR.get(case["overall_conclusion"])
    style = f"color:{color};font-weight:600" if color else "font-weight:600"
    badge_col2.markdown(
        f"**AI Conclusion:** <span style='{style}'>"
        f"{CONCLUSION_LABEL.get(case['overall_conclusion'], '—')}</span>",
        unsafe_allow_html=True,
    )

images = case["images"]
if not images:
    st.info("No images in this case.")
    st.stop()

reason_labels = [img["detection"]["predicted_label"] for img in images if img["detection"]]
reason_confidences = [img["detection"]["confidence_score"] for img in images if img["detection"]]
if reason_labels:
    reason_color = CONCLUSION_COLOR.get(case["overall_conclusion"])
    reason_text = conclusion_reason(reason_labels, reason_confidences)
    if reason_color:
        st.markdown(f"<span style='color:{reason_color}'>{reason_text}</span>", unsafe_allow_html=True)
    else:
        st.markdown(reason_text)

st.divider()

if "selected_image_idx" not in st.session_state:
    st.session_state.selected_image_idx = 0

if len(images) > 1:
    thumb_cols = st.columns(len(images))
    for i, img in enumerate(images):
        if thumb_cols[i].button(img["filename"], key=f"thumb_{img['id']}", use_container_width=True):
            st.session_state.selected_image_idx = i

idx = min(st.session_state.selected_image_idx, len(images) - 1)
selected = images[idx]
detection = selected["detection"]

left, right = st.columns(2)

with left:
    st.subheader(selected["filename"])
    show_overlay = st.radio("View", ["Original", "AI Overlay"], horizontal=True, key=f"view_{selected['id']}") == "AI Overlay"
    if show_overlay and detection:
        st.image(detection["overlay_path"], use_container_width=True)
        st.caption("🔴 Lichen &nbsp;&nbsp; 🟢 Other lesion &nbsp;&nbsp; ⬜ Normal", unsafe_allow_html=True)
    else:
        st.image(selected["image_path"], use_container_width=True)

with right:
    if detection:
        label_text, emoji = LABEL_DISPLAY[detection["predicted_label"]]
        st.markdown("### AI Detection Result")
        st.markdown(
            f"<div style='font-size:2rem;font-weight:700'>{emoji} {label_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No detection result for this image.")

    st.divider()
    st.markdown("### Clinical Notes")
    for note in case["notes"]:
        st.markdown(f"> {note['note_text']}")
        st.caption(note["created_at"][:19].replace("T", " "))

    draft = st.text_area("Add a note", placeholder="Clinical observations, differential diagnosis, follow-up plan…")
    if st.button("Save Note", disabled=not draft.strip()):
        add_note(case_id, draft.strip())
        st.rerun()

st.divider()
st.caption(
    "⚠️ This AI tool is a diagnostic assistant and does not substitute for a professional medical diagnosis."
)
