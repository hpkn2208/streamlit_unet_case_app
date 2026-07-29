"""Postgres (Supabase) + R2 persistence for patient cases — no auth, images in R2."""

import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import streamlit as st
from sqlalchemy import create_engine, text

import storage

CONCLUSION_MAP = {"lichen": "lichen_planus", "other_lesion": "other_lesion", "normal": "normal"}
PREDICTED_LABEL_DISPLAY = {"lichen": "Lichen Planus", "other_lesion": "Other Lesion", "normal": "Normal Mucosa"}
CONCLUSION_COLOR = {
    "lichen_planus": "#dc2626",
    "other_lesion": "#2ecc71",
    "normal": None,  # no override — inherits the theme's default text color (auto light/dark)
    "inconclusive": "#6b7280",
}


@lru_cache(maxsize=1)
def _engine():
    return create_engine(st.secrets["postgres_url"])


def init_db() -> None:
    with _engine().begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                case_code TEXT UNIQUE NOT NULL,
                patient_code TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'analyzed',
                overall_conclusion TEXT,
                created_by TEXT,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(text("ALTER TABLE cases ADD COLUMN IF NOT EXISTS created_by TEXT"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(id),
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_order INTEGER NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL UNIQUE REFERENCES images(id),
                yolo_detected INTEGER NOT NULL,
                yolo_boxes TEXT,
                lichen_pct REAL NOT NULL,
                other_pct REAL NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                overlay_path TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(id),
                note_text TEXT NOT NULL,
                author TEXT,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(text("ALTER TABLE notes ADD COLUMN IF NOT EXISTS author TEXT"))


def _generate_case_code(conn) -> str:
    year = datetime.now(timezone.utc).year
    count = conn.execute(
        text("SELECT COUNT(*) FROM cases WHERE case_code LIKE :pattern"),
        {"pattern": f"CASE-{year}-%"},
    ).scalar()
    return f"CASE-{year}-{count + 1:04d}"


# Clinical safety priority, most severe first: a single RELIABLE lichen-positive image makes
# the whole case lichen-positive, regardless of how many other images look normal. Same logic
# for other_lesion over normal. This is deliberately NOT a plain majority vote — diluting one
# positive finding against several negatives would risk masking a real lesion.
#
# "Reliable" means at least one occurrence of that label meets RELIABLE_CONFIDENCE_THRESHOLD.
# A single low-confidence call (e.g. 44% on a 3-class problem, barely above chance) must not
# be able to flip the case conclusion by itself — if nothing clears the bar, fall back to a
# plain majority vote and say so explicitly.
_SEVERITY_ORDER = ["lichen", "other_lesion", "normal"]
RELIABLE_CONFIDENCE_THRESHOLD = 0.5

_RULE_NOTE = {
    "lichen": "at least one image reliably shows lichen",
    "other_lesion": "at least one image reliably shows another lesion, none reliably show lichen",
    "normal": "all images are normal",
}


def _pick_winner(labels: list[str], confidences: list[float]) -> tuple[str, bool]:
    """Returns (winning_label, was_reliable)."""
    reliable_present = {
        label for label, conf in zip(labels, confidences) if conf >= RELIABLE_CONFIDENCE_THRESHOLD
    }
    for label in _SEVERITY_ORDER:
        if label in reliable_present:
            return label, True

    # Nothing cleared the confidence bar — fall back to plain majority vote, tie-broken by confidence.
    counts: dict[str, int] = {}
    best_conf: dict[str, float] = {}
    for label, conf in zip(labels, confidences):
        counts[label] = counts.get(label, 0) + 1
        best_conf[label] = max(best_conf.get(label, 0.0), conf)
    winner = max(counts, key=lambda l: (counts[l], best_conf[l]))
    return winner, False


def derive_overall_conclusion(labels: list[str], confidences: list[float]) -> str:
    """Most severe RELIABLE finding across all images wins: any lichen > any other lesion > all normal."""
    if not labels:
        return "inconclusive"
    winner, _ = _pick_winner(labels, confidences)
    return CONCLUSION_MAP[winner]


def conclusion_reason(labels: list[str], confidences: list[float]) -> str:
    """Human-readable explanation of how the case-level conclusion was reached."""
    if not labels:
        return "No images to base a conclusion on."

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    n_total = len(labels)
    parts = []
    for label in sorted(counts, key=lambda l: -counts[l]):
        n = counts[label]
        image_word = "image" if n == 1 else "images"
        parts.append(f"{n} of {n_total} {image_word} classified as {PREDICTED_LABEL_DISPLAY[label]}")

    winner, reliable = _pick_winner(labels, confidences)
    if reliable:
        rule_note = _RULE_NOTE[winner]
    else:
        rule_note = (
            f"no finding reached {int(RELIABLE_CONFIDENCE_THRESHOLD * 100)}% confidence — "
            f"falling back to the most common label; recommend manual review"
        )
    return "; ".join(parts) + f" → case conclusion: {PREDICTED_LABEL_DISPLAY[winner]} ({rule_note})."


def create_case(patient_code: str, images: list[dict], created_by: str | None = None) -> str:
    """images: list of {filename, image_rgb: np.ndarray, detection: dict from run_inference}."""
    labels = [img["detection"]["predicted_label"] for img in images]
    confidences = [img["detection"]["confidence_score"] for img in images]
    overall_conclusion = derive_overall_conclusion(labels, confidences)

    with _engine().begin() as conn:
        case_id = str(uuid.uuid4())
        case_code = _generate_case_code(conn)
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            text("""INSERT INTO cases (id, case_code, patient_code, status, overall_conclusion, created_by, created_at)
                    VALUES (:id, :case_code, :patient_code, :status, :overall_conclusion, :created_by, :created_at)"""),
            {"id": case_id, "case_code": case_code, "patient_code": patient_code,
             "status": "analyzed", "overall_conclusion": overall_conclusion, "created_by": created_by, "created_at": now},
        )

        for order, img in enumerate(images):
            image_id = str(uuid.uuid4())
            original_key = storage.upload_image(img["image_rgb"], f"{image_id}_original.png")
            overlay_key = storage.upload_image(img["detection"]["overlay_rgb"], f"{image_id}_overlay.png")

            conn.execute(
                text("""INSERT INTO images (id, case_id, filename, image_path, image_order)
                        VALUES (:id, :case_id, :filename, :image_path, :image_order)"""),
                {"id": image_id, "case_id": case_id, "filename": img["filename"],
                 "image_path": original_key, "image_order": order},
            )
            det = img["detection"]
            conn.execute(
                text("""INSERT INTO detections
                        (id, image_id, yolo_detected, yolo_boxes, lichen_pct, other_pct,
                         predicted_label, confidence_score, overlay_path, model_version)
                        VALUES (:id, :image_id, :yolo_detected, :yolo_boxes, :lichen_pct, :other_pct,
                                :predicted_label, :confidence_score, :overlay_path, :model_version)"""),
                {
                    "id": str(uuid.uuid4()), "image_id": image_id, "yolo_detected": int(det["yolo_detected"]),
                    "yolo_boxes": json.dumps(det["yolo_boxes"]), "lichen_pct": det["lichen_pct"],
                    "other_pct": det["other_pct"], "predicted_label": det["predicted_label"],
                    "confidence_score": det["confidence_score"], "overlay_path": overlay_key,
                    "model_version": det["model_version"],
                },
            )

    return case_id


def list_cases() -> list[dict]:
    with _engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT c.*, COUNT(i.id) AS image_count
            FROM cases c LEFT JOIN images i ON i.case_id = c.id
            GROUP BY c.id ORDER BY c.created_at DESC
        """)).mappings().all()
    return [dict(r) for r in rows]


def get_case(case_id: str) -> Optional[dict]:
    with _engine().connect() as conn:
        case = conn.execute(text("SELECT * FROM cases WHERE id = :id"), {"id": case_id}).mappings().first()
        if not case:
            return None
        case = dict(case)

        image_rows = conn.execute(
            text("SELECT * FROM images WHERE case_id = :id ORDER BY image_order"), {"id": case_id}
        ).mappings().all()
        images = []
        for img in image_rows:
            img = dict(img)
            det = conn.execute(
                text("SELECT * FROM detections WHERE image_id = :id"), {"id": img["id"]}
            ).mappings().first()
            det_dict = None
            if det:
                det_dict = dict(det)
                det_dict["yolo_detected"] = bool(det_dict["yolo_detected"])
                det_dict["yolo_boxes"] = json.loads(det_dict["yolo_boxes"]) if det_dict["yolo_boxes"] else None
                det_dict["overlay_path"] = storage.get_image_url(det_dict["overlay_path"])
            img["image_path"] = storage.get_image_url(img["image_path"])
            images.append({**img, "detection": det_dict})

        notes = conn.execute(
            text("SELECT * FROM notes WHERE case_id = :id ORDER BY created_at"), {"id": case_id}
        ).mappings().all()

    return {**case, "images": images, "notes": [dict(n) for n in notes]}


def add_note(case_id: str, note_text: str, author: str | None = None) -> None:
    with _engine().begin() as conn:
        note_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            text("INSERT INTO notes (id, case_id, note_text, author, created_at) VALUES (:id, :case_id, :note_text, :author, :created_at)"),
            {"id": note_id, "case_id": case_id, "note_text": note_text, "author": author, "created_at": now},
        )
        conn.execute(
            text("UPDATE cases SET status = 'reviewed' WHERE id = :id AND status = 'analyzed'"),
            {"id": case_id},
        )


def delete_all_cases() -> None:
    """Permanently removes every case, image, detection, and note (Postgres rows + R2 objects). Cannot be undone."""
    with _engine().begin() as conn:
        image_keys = conn.execute(text("SELECT image_path FROM images")).scalars().all()
        overlay_keys = conn.execute(text("SELECT overlay_path FROM detections")).scalars().all()
        conn.execute(text("DELETE FROM notes"))
        conn.execute(text("DELETE FROM detections"))
        conn.execute(text("DELETE FROM images"))
        conn.execute(text("DELETE FROM cases"))

    for key in [*image_keys, *overlay_keys]:
        storage.delete_image(key)
