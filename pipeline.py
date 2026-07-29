"""
YOLOv8-seg gate + 5-fold UNet ensemble inference pipeline.
Same models, thresholds, and logic as Stage3/streamlit_app/app.py, factored
into plain functions shared by the dashboard/new-case/case-detail pages.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
YOLO_PT = MODEL_DIR / "yolo_best.pt"
UNET_DIR = MODEL_DIR / "unet_folds"
N_FOLDS = 5
IMG_SIZE = 384
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PALETTE = {
    1: np.array([220, 50, 50], dtype=np.uint8),   # lichen — red
    2: np.array([50, 200, 80], dtype=np.uint8),   # other  — green
}

MODEL_VERSION = "yolo_best+unet5fold-v1"

_NORM = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
val_tf = A.Compose([_NORM, ToTensorV2()])
tta_tfs = [
    A.Compose([_NORM, ToTensorV2()]),
    A.Compose([A.HorizontalFlip(p=1), _NORM, ToTensorV2()]),
    A.Compose([A.VerticalFlip(p=1), _NORM, ToTensorV2()]),
]


@lru_cache(maxsize=1)
def load_yolo(path: str = str(YOLO_PT)) -> Optional[YOLO]:
    if not Path(path).exists():
        logger.warning("YOLO checkpoint not found: %s", path)
        return None
    return YOLO(path)


def _build_unet() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.5,
    )


@lru_cache(maxsize=2)
def load_unet_ensemble(unet_dir: str = str(UNET_DIR), n_folds: int = N_FOLDS, device_str: str = str(DEVICE)):
    device = torch.device(device_str)
    models = []
    for k in range(n_folds):
        ckpt = Path(unet_dir) / f"UNet_fold{k}_best.pth"
        if not ckpt.exists():
            logger.warning("UNet fold %d checkpoint not found: %s", k, ckpt)
            continue
        m = _build_unet().to(device)
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
        m.eval()
        models.append(m)
    return models


def remove_small_blobs(pred_mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 0:
        return pred_mask
    out = pred_mask.copy()
    lichen = (pred_mask == 1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lichen, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            out[labels == i] = 0
    return out


def predict_unet(models, img_rgb: np.ndarray, lichen_threshold: float, use_tta: bool, device: torch.device):
    """Return (pred H×W, probs C×H×W) from ensemble average."""
    tfs = tta_tfs if use_tta else [val_tf]
    probs_sum = None
    with torch.no_grad():
        for m in models:
            for i, tf in enumerate(tfs):
                t = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
                p = F.softmax(m(t), dim=1).squeeze(0).cpu()
                if i == 1:
                    p = p.flip(-1)  # undo hflip
                if i == 2:
                    p = p.flip(-2)  # undo vflip
                probs_sum = p if probs_sum is None else probs_sum + p
    probs = (probs_sum / (len(models) * len(tfs))).numpy()  # C×H×W

    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[probs[1] >= lichen_threshold] = 1
    other_m = (probs[2] > probs[0]) & (probs[2] > probs[1])
    pred[other_m & (pred == 0)] = 2
    return pred, probs


def draw_overlay(img_rgb: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img_rgb.copy()
    for cls, color in PALETTE.items():
        m = pred == cls
        if m.any():
            out[m] = (img_rgb[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    for cls, color in PALETTE.items():
        binary = (pred == cls).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color.tolist(), 2)
    return out


def yolo_crop(yolo_model: YOLO, img_bgr: np.ndarray, conf: float, padding: int, device=None):
    """Run YOLO on full image. Returns list of padded (x1,y1,x2,y2) bboxes, or None."""
    H, W = img_bgr.shape[:2]
    results = yolo_model(img_bgr, conf=conf, verbose=False, device=device)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    boxes = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(W, int(x2) + padding)
        y2 = min(H, int(y2) + padding)
        boxes.append((x1, y1, x2, y2))
    return boxes


def image_label_and_confidence(probs_by_crop: List[np.ndarray], full_pred: np.ndarray) -> Tuple[str, float]:
    """Derive a single predicted_label + confidence_score for the whole image."""
    lichen_frac = float((full_pred == 1).mean())
    other_frac = float((full_pred == 2).mean())

    if lichen_frac < 0.005 and other_frac < 0.005:
        label = "normal"
    elif lichen_frac >= other_frac:
        label = "lichen"
    else:
        label = "other_lesion"

    if probs_by_crop:
        cls_idx = {"normal": 0, "lichen": 1, "other_lesion": 2}[label]
        confidence = float(np.mean([float(p[cls_idx].max()) for p in probs_by_crop]))
    else:
        confidence = 0.5
    return label, confidence


def run_inference(
    img_rgb: np.ndarray,
    img_bgr: np.ndarray,
    use_yolo_gate: bool = True,
    yolo_conf: float = 0.15,
    yolo_padding: int = 40,
    lichen_thresh: float = 0.65,
    use_tta: bool = True,
    min_blob_px: int = 200,
):
    """Run the YOLO gate + UNet ensemble pipeline on one image.

    Retries on CPU if the GPU runs out of memory.
    """
    yolo_model = load_yolo()
    unet_models = load_unet_ensemble()
    if not unet_models:
        raise RuntimeError("No UNet checkpoints found under models/unet_folds/")

    try:
        return _run_inference_impl(
            yolo_model, unet_models, img_rgb, img_bgr,
            use_yolo_gate, yolo_conf, yolo_padding,
            lichen_thresh, use_tta, min_blob_px, DEVICE,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("GPU out of memory — retrying this image on CPU.")
        cpu_unet_models = load_unet_ensemble(str(UNET_DIR), N_FOLDS, "cpu")
        return _run_inference_impl(
            yolo_model, cpu_unet_models, img_rgb, img_bgr,
            use_yolo_gate, yolo_conf, yolo_padding,
            lichen_thresh, use_tta, min_blob_px, torch.device("cpu"),
            yolo_device="cpu",
        )


def _run_inference_impl(
    yolo_model, unet_models, img_rgb, img_bgr,
    use_yolo_gate, yolo_conf, yolo_padding,
    lichen_thresh, use_tta, min_blob_px, device,
    yolo_device=None,
):
    H, W = img_rgb.shape[:2]

    yolo_boxes = None
    if use_yolo_gate and yolo_model:
        yolo_boxes = yolo_crop(yolo_model, img_bgr, yolo_conf, yolo_padding, device=yolo_device)
    yolo_detected = yolo_boxes is not None

    full_pred = np.zeros((H, W), dtype=np.uint8)
    probs_by_crop = []

    if yolo_boxes:
        for (x1, y1, x2, y2) in yolo_boxes:
            crop_rgb = img_rgb[y1:y2, x1:x2]
            crop_256 = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
            pred_256, probs_256 = predict_unet(unet_models, crop_256, lichen_thresh, use_tta, device)
            pred_256 = remove_small_blobs(pred_256, min_blob_px)
            probs_by_crop.append(probs_256)

            pred_crop = cv2.resize(pred_256, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            full_pred[y1:y2, x1:x2] = np.maximum(full_pred[y1:y2, x1:x2], pred_crop)
    else:
        img_256 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        pred_256, probs_256 = predict_unet(unet_models, img_256, lichen_thresh, use_tta, device)
        pred_256 = remove_small_blobs(pred_256, min_blob_px)
        probs_by_crop.append(probs_256)
        full_pred = cv2.resize(pred_256, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay_rgb = draw_overlay(img_rgb, cv2.resize(full_pred, (W, H), interpolation=cv2.INTER_NEAREST))

    lichen_pct = float((full_pred == 1).mean() * 100)
    other_pct = float((full_pred == 2).mean() * 100)
    predicted_label, confidence_score = image_label_and_confidence(probs_by_crop, full_pred)

    return {
        "yolo_boxes": [list(b) for b in yolo_boxes] if yolo_boxes else None,
        "yolo_detected": yolo_detected,
        "overlay_rgb": overlay_rgb,
        "lichen_pct": lichen_pct,
        "other_pct": other_pct,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "model_version": MODEL_VERSION,
    }
