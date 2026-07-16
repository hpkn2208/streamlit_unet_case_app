"""SQLite persistence for patient cases — no auth, single-file DB, images on disk."""

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "cases.db"

CONCLUSION_MAP = {"lichen": "lichen_planus", "other_lesion": "other_lesion", "normal": "normal"}
PREDICTED_LABEL_DISPLAY = {"lichen": "Lichen Planus", "other_lesion": "Other Lesion", "normal": "Normal Mucosa"}
CONCLUSION_COLOR = {
    "lichen_planus": "#dc2626",
    "other_lesion": "#d97706",
    "normal": "#16a34a",
    "inconclusive": "#6b7280",
}


def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            case_code TEXT UNIQUE NOT NULL,
            patient_code TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'analyzed',
            overall_conclusion TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL REFERENCES cases(id),
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_order INTEGER NOT NULL
        );
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
        );
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL REFERENCES cases(id),
            note_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def _generate_case_code(conn: sqlite3.Connection) -> str:
    year = datetime.now(timezone.utc).year
    count = conn.execute(
        "SELECT COUNT(*) FROM cases WHERE case_code LIKE ?", (f"CASE-{year}-%",)
    ).fetchone()[0]
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
    conf_sum: dict[str, float] = {}
    for label, conf in zip(labels, confidences):
        counts[label] = counts.get(label, 0) + 1
        conf_sum[label] = conf_sum.get(label, 0.0) + conf

    n_total = len(labels)
    parts = []
    for label in sorted(counts, key=lambda l: -counts[l]):
        n = counts[label]
        avg_conf = conf_sum[label] / n * 100
        image_word = "image" if n == 1 else "images"
        conf_label = f"{avg_conf:.1f}% confidence" if n == 1 else f"avg {avg_conf:.1f}% confidence"
        low_conf_note = ", low confidence" if avg_conf < RELIABLE_CONFIDENCE_THRESHOLD * 100 else ""
        parts.append(f"{n} of {n_total} {image_word} classified as {PREDICTED_LABEL_DISPLAY[label]} ({conf_label}{low_conf_note})")

    winner, reliable = _pick_winner(labels, confidences)
    if reliable:
        rule_note = _RULE_NOTE[winner]
    else:
        rule_note = (
            f"no finding reached {int(RELIABLE_CONFIDENCE_THRESHOLD * 100)}% confidence — "
            f"falling back to the most common label; recommend manual review"
        )
    return "; ".join(parts) + f" → case conclusion: {PREDICTED_LABEL_DISPLAY[winner]} ({rule_note})."


def create_case(patient_code: str, images: list[dict]) -> str:
    """images: list of {filename, image_rgb: np.ndarray, detection: dict from run_inference}."""
    conn = _get_conn()
    case_id = str(uuid.uuid4())
    case_code = _generate_case_code(conn)
    now = datetime.now(timezone.utc).isoformat()

    labels = [img["detection"]["predicted_label"] for img in images]
    confidences = [img["detection"]["confidence_score"] for img in images]
    overall_conclusion = derive_overall_conclusion(labels, confidences)

    conn.execute(
        "INSERT INTO cases (id, case_code, patient_code, status, overall_conclusion, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (case_id, case_code, patient_code, "analyzed", overall_conclusion, now),
    )

    for order, img in enumerate(images):
        image_id = str(uuid.uuid4())
        original_path = IMAGES_DIR / f"{image_id}_original.png"
        overlay_path = IMAGES_DIR / f"{image_id}_overlay.png"
        Image.fromarray(img["image_rgb"]).save(original_path)
        Image.fromarray(img["detection"]["overlay_rgb"]).save(overlay_path)

        conn.execute(
            "INSERT INTO images (id, case_id, filename, image_path, image_order) VALUES (?, ?, ?, ?, ?)",
            (image_id, case_id, img["filename"], str(original_path), order),
        )
        det = img["detection"]
        conn.execute(
            """INSERT INTO detections
               (id, image_id, yolo_detected, yolo_boxes, lichen_pct, other_pct,
                predicted_label, confidence_score, overlay_path, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), image_id, int(det["yolo_detected"]), json.dumps(det["yolo_boxes"]),
                det["lichen_pct"], det["other_pct"], det["predicted_label"], det["confidence_score"],
                str(overlay_path), det["model_version"],
            ),
        )

    conn.commit()
    conn.close()
    return case_id


def list_cases() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT c.*, COUNT(i.id) AS image_count
           FROM cases c LEFT JOIN images i ON i.case_id = c.id
           GROUP BY c.id ORDER BY c.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_case(case_id: str) -> Optional[dict]:
    conn = _get_conn()
    case = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not case:
        conn.close()
        return None

    image_rows = conn.execute(
        "SELECT * FROM images WHERE case_id = ? ORDER BY image_order", (case_id,)
    ).fetchall()
    images = []
    for img in image_rows:
        det = conn.execute("SELECT * FROM detections WHERE image_id = ?", (img["id"],)).fetchone()
        det_dict = None
        if det:
            det_dict = dict(det)
            det_dict["yolo_detected"] = bool(det_dict["yolo_detected"])
            det_dict["yolo_boxes"] = json.loads(det_dict["yolo_boxes"]) if det_dict["yolo_boxes"] else None
        images.append({**dict(img), "detection": det_dict})

    notes = conn.execute(
        "SELECT * FROM notes WHERE case_id = ? ORDER BY created_at", (case_id,)
    ).fetchall()

    conn.close()
    return {**dict(case), "images": images, "notes": [dict(n) for n in notes]}


def add_note(case_id: str, note_text: str) -> None:
    conn = _get_conn()
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO notes (id, case_id, note_text, created_at) VALUES (?, ?, ?, ?)",
        (note_id, case_id, note_text, now),
    )
    conn.execute(
        "UPDATE cases SET status = 'reviewed' WHERE id = ? AND status = 'analyzed'", (case_id,)
    )
    conn.commit()
    conn.close()


def delete_all_cases() -> None:
    """Permanently removes every case, image, detection, and note. Cannot be undone."""
    conn = _get_conn()
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM detections")
    conn.execute("DELETE FROM images")
    conn.execute("DELETE FROM cases")
    conn.commit()
    conn.close()

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(exist_ok=True)
