import cv2
import numpy as np
import streamlit as st
from PIL import Image

from db import create_case
from pipeline import MODEL_CHOICES, run_inference

st.set_page_config(page_title="New Case — Lichen Detection", layout="wide")

st.title("New Case")
st.caption("Upload one or more clinical images for a patient to run lesion detection.")

if st.button("← Back to dashboard"):
    st.switch_page("Dashboard.py")

patient_code = st.text_input("Patient code", placeholder="e.g. PT-0742")
st.caption("Internal alias only — no patient name or DOB is stored.")

uploaded_files = st.file_uploader(
    "Upload oral images (PNG / JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

model_choice_label = st.radio(
    "Segmentation model",
    options=list(MODEL_CHOICES.values()),
    index=0,
    horizontal=True,
)
model_choice = next(k for k, v in MODEL_CHOICES.items() if v == model_choice_label)

with st.expander("Inference settings (defaults work well)"):
    yolo_conf = st.slider("YOLO confidence threshold", 0.05, 0.50, 0.15, 0.05)
    yolo_padding = st.slider("YOLO crop padding (px)", 0, 100, 40, 10)
    lichen_thresh = st.slider("Lichen probability threshold", 0.30, 0.90, 0.75, 0.05)
    min_blob_px = st.slider("Min lesion blob (pixels)", 0, 1000, 500, 50)
    use_tta = st.checkbox("Test-time augmentation (TTA)", value=True)
    use_yolo_gate = st.checkbox("Enable YOLO gate", value=True)

can_analyze = bool(uploaded_files) and bool(patient_code.strip())

if st.button("Analyze", type="primary", disabled=not can_analyze):
    images_payload = []
    progress = st.progress(0.0, text=f"Running YOLO gate + {model_choice_label}…")

    for i, uf in enumerate(uploaded_files):
        img_pil = Image.open(uf).convert("RGB")
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            detection = run_inference(
                img_rgb, img_bgr,
                use_yolo_gate=use_yolo_gate, yolo_conf=yolo_conf, yolo_padding=yolo_padding,
                lichen_thresh=lichen_thresh, use_tta=use_tta, min_blob_px=min_blob_px,
                model_choice=model_choice,
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        images_payload.append({"filename": uf.name, "image_rgb": img_rgb, "detection": detection})
        progress.progress((i + 1) / len(uploaded_files), text=f"Analyzed {i + 1}/{len(uploaded_files)}")

    case_id = create_case(patient_code.strip(), images_payload)
    st.session_state["current_case_id"] = case_id
    st.switch_page("pages/2_Case_Detail.py")
