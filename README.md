# KARSL-502 Sign Language Recognition

Real-time Arabic Sign Language Recognition system trained on the **KARSL-502 dataset** (502 signs, 3 signers) using **CTR-GCN** with 3-stream score fusion and **MediaPipe Holistic** for landmark extraction.

---

## Results

| Stream | Top-1 | Top-5 |
|--------|-------|-------|
| Joint  | 73.6% | 90.2% |
| Bone   | 79.9% | 91.8% |
| Motion | 53.7% | 71.9% |
| **Fused** | **86.94%** | **95.10%** |

Evaluated on an unseen signer (signer-independent test set).

---

## Architecture

- **Backbone:** CTR-GCN (Channel-wise Topology Refinement Graph Convolution)
- **Head:** Temporal Transformer (2 layers, 4 heads)
- **Streams:** Joint + Bone + Motion — fused via weighted score averaging (0.35 / 0.40 / 0.25)
- **Skeleton:** 75 joints — Face Mesh (33) + Left Hand (21) + Right Hand (21)
- **Frames:** 32 frames per clip
- **Classes:** 502 Arabic signs

---


## Pipeline

```
Raw videos/images
      ↓
extract_landmarks.py                →   .npz files (543 landmarks, hip-normalized)
      ↓
CTRGCN_input.py                     →   train_data.npy / eval_data.npy  (3, 32, 75, 1)
      ↓
train_ctrgcn_v2.py                  →   best_model.pth
      ↓
CTR_GCN_NLP_cam_inference.py        →   real-time inference with LLM-based language refinement
```

---

## Setup

```bash
pip install torch torchvision mediapipe opencv-python numpy pandas scikit-learn tqdm openpyxl arabic-reshaper python-bidi Pillow
```

---

## Usage

**Step 1 — Extract landmarks from your dataset:**
```bash
python core/src/data_prep/extract_landmarks.py
```

**Step 2 — Prepare training input:**
```bash
python core/src/data_prep/2AGCN_input.py
```

**Step 3 — Train:**
```bash
python core/src/trial_final/train_ctrgcn_v6.py
```

**Step 4 — Real-time camera test:**
```bash
python core/src/trial_final/camera_v6.py \
    --checkpoint "core/output/checkpoints_v6/best_model.pth" \
    --labels     "core/src/trial_final/KARSL-502_Labels.xlsx"
```

Optional flags:
```
--camera      int    Camera index (default: 0)
--complexity  0|1|2  MediaPipe model complexity (default: 2)
--threshold   float  Confidence threshold (default: 0.35)
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD + Nesterov |
| Base LR | 0.05 |
| Schedule | Cosine warmup (5 ep) → decay → SWA (ep 60–80) |
| SWA LR | 5e-5 (fixed) |
| Batch size | 32 |
| Epochs | 80 |
| Label smoothing | 0.1 |
| Dropout | 0.5 |
| Weight decay | 5e-4 |

**Augmentations:** speed warp, signer-style aug (global translation + per-region scale), spatial rotation, hand dropout, face dropout, z-axis noise, temporal jitter.

---

## Dataset

**KARSL-502** — Kuwait Arabic Sign Language dataset.
- 502 sign classes
- 3 signers (2 train, 1 test — signer-independent evaluation)
- ~50k training samples / ~24k test samples

Dataset is not included in this repo. Contact the dataset authors for access.

---

## Notes

- The model uses **signer-independent evaluation** — the test signer never appears in training.
- Landmark normalization is relative to the **mid-hip point** from MediaPipe Pose, with a fallback to nose-tip if pose is not detected.
- The `bone_pairs` used for the Bone stream are saved inside `best_model.pth` and loaded automatically at inference time.
