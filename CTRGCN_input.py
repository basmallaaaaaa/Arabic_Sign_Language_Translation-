import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
SOURCE_DIR = r"C:\KARSL-502\core\output\data\full_kps"
OUTPUT_DIR = r"C:\KARSL-502\core\output\data\agcn_final_input"

# ✅ FIX: 32 فريم بدل 24 — الموديل بيتدرب على 32 فريم
# (كان 24 قبل، وده كان بيسبب double-resampling: 24→32 في الـ training augmentation)
TARGET_FRAMES = 32
NUM_JOINTS    = 75
SELECTED_INDICES = list(range(0, 33)) + list(range(501, 543))


def process_single_file(task_info):
    file_path, is_train, class_id = task_info
    results = []

    try:
        data = np.load(file_path)
        for sample_key in data.keys():
            raw_seq = data[sample_key]  # (T, 543, 4)

            assert raw_seq.ndim == 3,          f"Expected 3D array, got {raw_seq.ndim}D"
            assert raw_seq.shape[1] == 543,    f"Expected 543 landmarks, got {raw_seq.shape[1]}"
            assert raw_seq.shape[2] == 4,      f"Expected 4 channels, got {raw_seq.shape[2]}"

            T_actual = raw_seq.shape[0]
            if T_actual < 8:
                print(f"⚠️  Skipping {sample_key}: only {T_actual} frames")
                continue

            # ── filter landmarks: 33 face-mesh + 21 LH + 21 RH = 75 joints ──
            skeleton = raw_seq[:, SELECTED_INDICES, :3]   # (T, 75, 3)
            # ⚠️  NO extra normalization here!
            # الـ _norm.npz متنورمالايز أصلاً بالـ mid-hip في extract_landmarks.py
            # أي normalization إضافية = double normalization = data خربانة

            # ── resize to TARGET_FRAMES ──────────────────────────────────
            seq_tensor = (torch.from_numpy(skeleton)
                          .permute(2, 0, 1)          # (3, T, 75)
                          .unsqueeze(0)               # (1, 3, T, 75)
                          .float())
            resized = F.interpolate(
                seq_tensor,
                size=(TARGET_FRAMES, skeleton.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).numpy()                     # (3, 32, 75)

            final_seq = np.expand_dims(resized, axis=-1)   # (3, 32, 75, 1)
            results.append((final_seq, class_id, is_train))

    except Exception as e:
        print(f"❌ Error in {file_path}: {e}")
        return []

    return results


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tasks   = []
    folders = [f for f in os.listdir(SOURCE_DIR)
               if os.path.isdir(os.path.join(SOURCE_DIR, f))]

    for folder in folders:
        signer_id = folder.split('-')[0]
        is_train  = signer_id in ['01', '02']
        folder_path = os.path.join(SOURCE_DIR, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('_norm.npz')]

        for file in files:
            try:
                class_id = int(file.split('_')[0]) - 1
                tasks.append((os.path.join(folder_path, file), is_train, class_id))
            except Exception:
                continue

    train_data, train_labels = [], []
    eval_data,  eval_labels  = [], []

    max_threads = 8
    print(f"Hardware Optimization: Using ThreadPool with {max_threads} workers.")
    print(f"Processing {len(tasks)} files...  (TARGET_FRAMES={TARGET_FRAMES})")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_task = {executor.submit(process_single_file, task): task
                          for task in tasks}

        for future in tqdm(as_completed(future_to_task),
                           total=len(tasks), desc="Processing Landmarks"):
            for seq, label, is_train_flag in future.result():
                if is_train_flag:
                    train_data.append(seq);  train_labels.append(label)
                else:
                    eval_data.append(seq);   eval_labels.append(label)

    print("\nFinalizing datasets. Saving large .npy files...")
    np.save(os.path.join(OUTPUT_DIR, "train_data.npy"),
            np.array(train_data,  dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "train_label.npy"),
            np.array(train_labels, dtype=np.int16))
    np.save(os.path.join(OUTPUT_DIR, "eval_data.npy"),
            np.array(eval_data,   dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "eval_label.npy"),
            np.array(eval_labels,  dtype=np.int16))

    print(f"\n✅ Done. Files saved to: {OUTPUT_DIR}")
    print(f"   Train: {len(train_data):,} samples  |  Eval: {len(eval_data):,} samples")
    print(f"   Shape per sample: {train_data[0].shape if train_data else 'N/A'}")


if __name__ == "__main__":
    main()