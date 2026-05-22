import os
import csv
import time
import logging
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Logging Setup
LOG_FILE = "extraction_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Landmark Extraction
# ─────────────────────────────────────────────
def get_all_landmarks(results):
    """
    Extract 543 landmarks:
      - Face:       468 points  (indices 0   → 467)
      - Pose:        33 points  (indices 468 → 500)
      - Left Hand:   21 points  (indices 501 → 521)
      - Right Hand:  21 points  (indices 522 → 542)

    Each point has 4 channels: [x, y, z, visibility]
    visibility is 0.0 for face/hands (not provided by MediaPipe)
    visibility is the real confidence score for pose landmarks
    """
    # Face (468 points) — no visibility score in MediaPipe
    face = (
        np.array([[lm.x, lm.y, lm.z, 0.0] for lm in results.face_landmarks.landmark])
        if results.face_landmarks else np.zeros((468, 4))
    )

    # Pose (33 points) — has visibility score
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        if results.pose_landmarks else np.zeros((33, 4))
    )

    # Left Hand (21 points)
    lh = (
        np.array([[lm.x, lm.y, lm.z, 0.0] for lm in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks else np.zeros((21, 4))
    )

    # Right Hand (21 points)
    rh = (
        np.array([[lm.x, lm.y, lm.z, 0.0] for lm in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks else np.zeros((21, 4))
    )

    # Final shape: (543, 4)
    return np.vstack([face, pose, lh, rh])


def normalize_landmarks(landmarks):
    """
    Normalize all landmarks relative to the mid-hip point.
    - Pose mid-hip = average of left_hip (468+23) and right_hip (468+24)
    - Subtracting the root makes the model translation-invariant.
    - Only x, y, z are normalized (channel 4 is visibility, left as-is)
    """
    left_hip  = landmarks[468 + 23, :3]
    right_hip = landmarks[468 + 24, :3]
    mid_hip   = (left_hip + right_hip) / 2.0

    normalized = landmarks.copy()
    normalized[:, :3] -= mid_hip   # shift x,y,z only
    return normalized


# ─────────────────────────────────────────────
# Per-Sign Extraction (one sign_id at a time)
# ─────────────────────────────────────────────
def extract_single_sign(args):
    """
    Extracts landmarks for one sign folder and saves two NPZ files:
      - <sign_id>.npz       → raw coordinates
      - <sign_id>_norm.npz  → normalized coordinates (relative to mid-hip)

    Designed to be called either directly or via multiprocessing Pool.
    """
    sign_full_path, save_dir, sign_id = args

    output_file      = os.path.join(save_dir, f"{sign_id}.npz")
    output_file_norm = os.path.join(save_dir, f"{sign_id}_norm.npz")

    # Skip if both files already exist
    if os.path.exists(output_file) and os.path.exists(output_file_norm):
        return {"sign_id": sign_id, "status": "skipped", "samples": 0, "empty_sequences": 0}

    samples = sorted([
        d for d in os.listdir(sign_full_path)
        if os.path.isdir(os.path.join(sign_full_path, d))
    ])

    output_dict      = {}   # raw
    output_dict_norm = {}   # normalized

    empty_sequence_count = 0

    # Each subprocess creates its own MediaPipe instance
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for sample_name in tqdm(samples, desc=f"Sign {sign_id}", leave=False):
            sample_path = os.path.join(sign_full_path, sample_name)
            images = sorted([
                i for i in os.listdir(sample_path)
                if i.lower().endswith(('.jpg', '.jpeg'))
            ])

            sequence_data      = []
            sequence_data_norm = []

            for img_name in images:
                try:
                    img = cv2.imread(os.path.join(sample_path, img_name))
                    if img is None:
                        logger.warning(f"⚠️  Could not read image: {img_name} in {sample_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = holistic.process(img_rgb)

                    raw_lm  = get_all_landmarks(results)
                    norm_lm = normalize_landmarks(raw_lm)

                    sequence_data.append(raw_lm)
                    sequence_data_norm.append(norm_lm)

                except Exception as e:
                    logger.error(f"❌ Error processing frame {img_name} in {sample_path}: {e}")
                    continue

            if sequence_data:
                output_dict[sample_name]      = np.array(sequence_data)       # shape: (T, 543, 4)
                output_dict_norm[sample_name] = np.array(sequence_data_norm)  # shape: (T, 543, 4)
            else:
                empty_sequence_count += 1
                logger.warning(f"⚠️  Empty sequence — no valid frames for sample: {sample_name} | Sign: {sign_id}")

    # Save compressed NPZ files
    if output_dict:
        np.savez_compressed(output_file,      **output_dict)
        np.savez_compressed(output_file_norm, **output_dict_norm)

    return {
        "sign_id":         sign_id,
        "status":          "done",
        "samples":         len(samples),
        "empty_sequences": empty_sequence_count
    }


# ─────────────────────────────────────────────
# CSV Stats Logger
# ─────────────────────────────────────────────
def log_stats_to_csv(csv_path, signer, split, stats_list):
    """Appends extraction stats for each sign to a CSV for later analysis."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "signer", "split", "sign_id", "status", "samples", "empty_sequences"])
        if not file_exists:
            writer.writeheader()
        for stat in stats_list:
            writer.writerow({
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signer":          signer,
                "split":           split,
                **stat
            })


# ─────────────────────────────────────────────
# Main Extraction Loop
# ─────────────────────────────────────────────
def run_global_extraction(input_root, output_root, use_multiprocessing=False, num_workers=2):
    """
    Main driver function.

    Args:
        input_root          : Root folder containing signer subdirectories.
        output_root         : Where to save NPZ files.
        use_multiprocessing : Set True to parallelize across sign folders.
                              ⚠️  Keep False on low-RAM machines (<16GB).
        num_workers         : Number of parallel workers (used only if above is True).
                              Recommended: 2 (MediaPipe is already multi-threaded internally).
    """
    os.makedirs(output_root, exist_ok=True)
    stats_csv = os.path.join(output_root, "extraction_stats.csv")

    signers = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])

    total_start = time.time()

    for signer in signers:
        signer_path = os.path.join(input_root, signer)

        for split in ["train", "test"]:
            current_split_path = os.path.join(signer_path, split)
            if not os.path.exists(current_split_path):
                continue

            save_dir = os.path.join(output_root, f"{signer}-{split}")
            os.makedirs(save_dir, exist_ok=True)

            sign_folders = sorted(os.listdir(current_split_path))
            logger.info(f"\n📂 Signer: {signer} | Split: {split.upper()} | Signs: {len(sign_folders)}")

            # Build task list: (sign_full_path, save_dir, sign_id)
            tasks = [
                (os.path.join(current_split_path, sign_id), save_dir, sign_id)
                for sign_id in sign_folders
            ]

            if use_multiprocessing:
                workers = min(num_workers, cpu_count())
                logger.info(f"⚡ Using multiprocessing with {workers} workers")
                with Pool(processes=workers) as pool:
                    stats_list = pool.map(extract_single_sign, tasks)
            else:
                stats_list = [extract_single_sign(task) for task in tasks]

            log_stats_to_csv(stats_csv, signer, split, stats_list)

            done    = sum(1 for s in stats_list if s["status"] == "done")
            skipped = sum(1 for s in stats_list if s["status"] == "skipped")
            empty   = sum(s["empty_sequences"] for s in stats_list)
            logger.info(f"✅ Done: {done} | Skipped: {skipped} | Empty sequences: {empty}")

    elapsed = time.time() - total_start
    logger.info(f"\n🏁 MISSION ACCOMPLISHED — Total time: {elapsed/60:.1f} minutes")
    logger.info(f"📊 Stats saved to: {stats_csv}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
INPUT_DIR  = r"C:\KARSL-502"
OUTPUT_DIR = r"C:\KARSL-502\core\output\data\full_kps"

if __name__ == "__main__":
    run_global_extraction(
        input_root          = INPUT_DIR,
        output_root         = OUTPUT_DIR,
        use_multiprocessing = False,   # Set True for speed boost (needs ~16GB RAM)
        num_workers         = 2
    )
