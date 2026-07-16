# 🦷 Lichen Case Dashboard

**AI-assisted patient case tracking for oral lichen planus screening — YOLOv8 lesion detection + dual UNet segmentation ensembles, in a single deployable Streamlit app.**

No login, no database server, no Docker. Clone it, drop in your model weights, and it runs.

> ⚠️ **This is a diagnostic assistant, not a diagnosis.** All output must be reviewed by a qualified clinician.

---

## What it does

Upload one or more intraoral photos for a patient, and the app runs a two-stage pipeline — a YOLOv8-seg lesion detector gates the crop, then a segmentation ensemble classifies every pixel as **lichen planus**, **other lesion**, or **normal mucosa**. Results are saved as a case you can revisit, annotate, and re-analyze.

| | |
|---|---|
| 🖼️ **Multi-image cases** | Upload a full set of photos per patient in one pass |
| 🧠 **Two swappable ensembles** | Plain 5-fold UNet *or* a 5-fold Attention UNet (UNet++ with scSE attention gates) — pick per analysis |
| 🔁 **Re-analyze on demand** | Already ran with one ensemble? Re-run the same case through the other with one click, no re-upload |
| 🗳️ **Explainable conclusions** | Case-level verdict is a majority vote across images, with the vote breakdown and per-class confidence shown in plain language |
| 🎨 **Color-coded overlays** | Red / amber pixel overlays on the source image, toggleable against the original |
| 📝 **Clinical notes** | Timestamped free-text notes per case |
| ⚡ **Warm start** | Every model loads once when the app boots, so the first analysis isn't stuck behind a cold load |
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
        │   Plain UNet   OR   Attention UNet   │  5-fold ensemble + TTA
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
        case-level conclusion = majority vote across all
              images in the case (ties broken by confidence)
```

Both segmentation architectures share the same YOLO gate, encoder backbone, input resolution, and test-time augmentation — the only difference is the decoder (plain UNet vs. UNet++ with [scSE](https://arxiv.org/abs/1803.02579) attention blocks), so results are directly comparable.

---

## Screens

- **Dashboard** — case list with live stats (total / pending / lichen-positive), one click into any case
- **New Case** — patient code + drag-and-drop multi-image upload, ensemble picker, adjustable inference thresholds
- **Case Detail** — per-image overlay/original toggle, confidence + area breakdown, conclusion reasoning, notes, and the re-analyze control

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
    ├── UNet_fold0_best.pth  …  UNet_fold4_best.pth              (plain UNet, 5-fold CV)
    └── UNet++_+_scSE_fold0_best.pth  …  UNet++_+_scSE_fold4_best.pth   (attention UNet, 5-fold CV)
```

Both UNet variants expect an EfficientNet-B0 encoder, 3-class output (normal / lichen / other), 256×256 input. Missing a few folds is fine — the ensemble just averages over whatever it finds. If an entire ensemble is missing, selecting it in the model picker will show a clear error rather than fail silently.

---

## Deploy for free (Streamlit Community Cloud)

No Docker, no server to manage:

1. Push this repo to GitHub (weights included — a full set of both ensembles is ~270 MB, well under GitHub's limits)
2. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**
3. Point it at this repo, branch `main`, main file `Dashboard.py`
4. Deploy

Runs on CPU on the free tier — a few seconds per image instead of ~1s on GPU. No code changes needed; the pipeline already has an automatic CUDA→CPU fallback.

---

## Project structure

```
streamlit_case_app/
├── Dashboard.py              # entry point — case list, stats, eager model load, danger zone
├── pages/
│   ├── 1_New_Case.py         # upload + ensemble choice + run inference
│   └── 2_Case_Detail.py      # results, overlay toggle, notes, re-analyze
├── pipeline.py                # YOLO gate + dual UNet ensembles, framework-agnostic
├── db.py                     # SQLite persistence — cases, images, detections, notes
├── models/                   # your trained weights (see above)
├── requirements.txt
├── packages.txt               # apt packages Streamlit Cloud needs (libGL, etc.)
└── data/                     # created at runtime — SQLite DB + saved images (gitignored)
```

---

## Data & privacy

- Cases are keyed by a doctor-assigned **patient code**, not a name or date of birth
- All data lives in `data/` — a local SQLite file plus original/overlay PNGs — created on first run and excluded from git
- On Streamlit Community Cloud, `data/` does **not** survive a redeploy or app restart — this is a pilot/demo tool, not a system of record
- No accounts, no authentication — anyone with the app URL can view and create cases

---

## Disclaimer

This tool assists with screening and triage. It does not replace clinical judgment, biopsy, or a licensed diagnosis. Always confirm AI output with a qualified oral medicine specialist.
