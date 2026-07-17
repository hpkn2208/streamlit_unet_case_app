import cv2
import numpy as np
import streamlit as st
from PIL import Image

from db import create_case
from pipeline import run_inference

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

YOLO_CONF = 0.15
YOLO_PADDING = 40
LICHEN_THRESH = 0.65
USE_TTA = True
USE_YOLO_GATE = True
MIN_BLOB_PX = 200

can_analyze = bool(uploaded_files) and bool(patient_code.strip())

if st.button("Analyze", type="primary", disabled=not can_analyze):
    images_payload = []
    progress = st.progress(0.0, text="Running YOLO gate + 5-fold UNet ensemble…")

    for i, uf in enumerate(uploaded_files):
        img_pil = Image.open(uf).convert("RGB")
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            detection = run_inference(
                img_rgb, img_bgr,
                use_yolo_gate=USE_YOLO_GATE, yolo_conf=YOLO_CONF, yolo_padding=YOLO_PADDING,
                lichen_thresh=LICHEN_THRESH, use_tta=USE_TTA, min_blob_px=MIN_BLOB_PX,
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        images_payload.append({"filename": uf.name, "image_rgb": img_rgb, "detection": detection})
        progress.progress((i + 1) / len(uploaded_files), text=f"Analyzed {i + 1}/{len(uploaded_files)}")

    case_id = create_case(patient_code.strip(), images_payload)
    st.session_state["current_case_id"] = case_id
    st.switch_page("pages/2_Case_Detail.py")
