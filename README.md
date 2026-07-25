# 🦷 Lichen Case Dashboard

**AI-assisted patient case tracking for oral lichen planus screening — YOLOv8 lesion detection + a 5-fold UNet segmentation ensemble, in a single deployable Streamlit app.**

No login, no database server, no Docker. Clone it, drop in your model weights, and it runs.

> ⚠️ **This is a diagnostic assistant, not a diagnosis.** All output must be reviewed by a qualified clinician.

---

## What it does

Upload one or more intraoral photos for a patient, and the app runs a two-stage pipeline — a YOLOv8-seg lesion detector gates the crop, then a 5-fold UNet ensemble classifies every pixel as **lichen planus**, **other lesion**, or **normal mucosa**. Results are saved as a case you can revisit and annotate.

| | |
|---|---|
| 🖼️ **Multi-image cases** | Upload a full set of photos per patient in one pass |
| 🧠 **5-fold UNet ensemble** | EfficientNet-B0 encoder, softmax-averaged across folds + test-time augmentation |
| 🩺 **Safety-first conclusions** | Case-level verdict prioritizes any *reliable* positive finding over normal readings elsewhere — a single confident lichen call isn't diluted by other normal images. Low-confidence findings (<50%) are explicitly flagged and can't decide the case alone |
| 🎨 **Color-coded overlays** | Red / amber pixel overlays on the source image, toggleable against the original |
| 📝 **Clinical notes** | Timestamped free-text notes per case |
| ⚡ **Warm start** | The model loads once when the app boots, so the first analysis isn't stuck behind a cold load |
| 🗑️ **One-click reset** | Guarded "danger zone" control to wipe all demo/pilot data |

---

## How it works

```
                 ┌──────────────┐
   image(s)  ──▶ │  YOLOv8-seg  │  gate: crop to the lesion ROI (falls back to
                 └──────┬───────┘         full-frame if nothing is detected)
                        │
                        ▼
        ┌───────────────────────────────┐
        │     5-fold UNet ensemble       │  softmax-averaged across folds + TTA
        │  (EfficientNet-B0 encoder)     │  (horizontal/vertical flip averaging)
        └───────────────┬───────────────┘
                        │
                        ▼
        per-pixel softmax → lichen / other / normal
                        │
                        ▼
   overlay + area %  +  per-image label  +  confidence
                        │
                        ▼
   case-level conclusion: most severe RELIABLE finding wins
   (lichen > other lesion > normal — each requires ≥50% confidence
    in at least one image, else falls back to majority vote)
```

---

## Screens

- **Dashboard** — case list with live stats (total / pending / lichen-positive), one click into any case
- **New Case** — patient code + drag-and-drop multi-image upload, adjustable inference thresholds
- **Case Detail** — per-image overlay/original toggle, confidence + area breakdown, conclusion reasoning, notes

---

## Quickstart

```bash
git clone <this-repo>
cd streamlit_case_app
pip install -r requirements.txt
```

Drop your trained checkpoints into `models/` (see [Model weights](#model-weights) below), then:

```bash
streamlit run Dashboard.py
```

Open `http://localhost:8501`.

---

## Model weights

Not included in this repo — supply your own trained checkpoints in this layout:

```
models/
├── yolo_best.pt
└── unet_folds/
    └── UNet_fold0_best.pth  …  UNet_fold4_best.pth   (5-fold cross-validation)
```

Expects an EfficientNet-B0 encoder, 3-class output (normal / lichen / other), 256×256 input. Missing a few folds is fine — the ensemble just averages over whatever it finds.

---

## Deploy for free (Streamlit Community Cloud)

No Docker, no server to manage:

1. Push this repo to GitHub (weights included — a full set is ~140 MB, well under GitHub's limits)
2. Create a free [Supabase](https://supabase.com) project (Postgres) and a free [Cloudflare R2](https://developers.cloudflare.com/r2/) bucket — see [Persistent storage](#persistent-storage) below
3. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**
4. Point it at this repo, branch `main`, **main file path: `Dashboard.py`** (not `pipeline.py` — that's just the inference logic, it has no UI)
5. In app **Settings → Secrets**, paste the keys from `.streamlit/secrets.toml.example` filled in with your real Supabase/R2 values
6. Deploy

Runs on CPU on the free tier — a few seconds per image instead of ~1s on GPU. No code changes needed; the pipeline already has an automatic CUDA→CPU fallback.

---

## Persistent storage

Streamlit Community Cloud has no persistent disk — its filesystem is rebuilt from git on every redeploy or wake-from-sleep, which used to wipe `data/` (SQLite db + saved images) after about a day of inactivity. This app now stores everything externally instead:

- **Images** → Cloudflare R2 (S3-compatible object storage, free 10GB, no egress fees) via `storage.py`
- **Case/image/detection/note records** → Supabase Postgres (free 500MB, always-on) via `db.py`

Set up:
1. Supabase: new project → Project Settings → Database → copy the connection string
2. Cloudflare R2: create a bucket → create an API token scoped to that bucket (Object Read & Write) → note the Account ID, Access Key ID, Secret Access Key
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in both (gitignored, never commit real secrets)
4. `migrations/001_init.sql` documents the schema `db.py::init_db()` creates automatically on first run — run it manually in the Supabase SQL editor only if you want the tables to exist before the app's first boot

---

## Project structure

```
streamlit_case_app/
├── Dashboard.py              # entry point — case list, stats, eager model load, danger zone
├── pages/
│   ├── 1_New_Case.py         # upload + run inference
│   └── 2_Case_Detail.py      # results, overlay toggle, notes
├── pipeline.py                # YOLO gate + UNet ensemble, framework-agnostic
├── db.py                     # Postgres (Supabase) persistence — cases, images, detections, notes
├── storage.py                 # Cloudflare R2 image upload/download
├── migrations/001_init.sql    # reference schema (db.py also creates it automatically)
├── models/                   # your trained weights (see above)
├── requirements.txt
├── packages.txt               # apt packages Streamlit Cloud needs (libGL, etc.)
└── .streamlit/secrets.toml.example  # copy to secrets.toml and fill in (gitignored)
```

---

## Data & privacy

- Cases are keyed by a doctor-assigned **patient code**, not a name or date of birth
- Images live in Cloudflare R2, case/note records live in Supabase Postgres — both persist across redeploys and app restarts (see [Persistent storage](#persistent-storage))
- No accounts, no authentication — anyone with the app URL can view and create cases

---

## Disclaimer

This tool assists with screening and triage. It does not replace clinical judgment, biopsy, or a licensed diagnosis. Always confirm AI output with a qualified oral medicine specialist.
