-- Case-tracking schema. Also created automatically by db.py::init_db() on
-- first run — this file is a reference for running directly in the
-- Supabase SQL editor if you want the tables to exist before first boot.

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
