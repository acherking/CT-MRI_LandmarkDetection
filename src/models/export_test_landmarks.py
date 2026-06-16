"""
Reconstruct the held-out test-set predicted landmarks from cropped voxel space back to
full-volume (original-scan) voxel coordinates, persist them, and emit per-patient aug_1 CSVs.

Usage:  python export_test_landmarks.py [ct|mr]      (default: ct)

Test set (both modalities): final_Y_test_pred.npy / Y_test_true.npy, shape (400, 2, 3)
    = patients [GM, GE, AZ, LP] x 2 ears x 50 augmentations (same static data split).

The model's eval path (load_dataset_manager) never saves a per-instance crop offset, so we
rebuild it from the source crop dataset's length array, apply the same cut_layers adjustment
as load_dataset_crop, take the test split, and reconstruct (un-flip the right ear on the
column axis, then add length).

MR only: the .mat grid is isotropic (0.2604) while the original MRI is anisotropic
(0.2604, 0.2604, 0.3). x,y match the original exactly; z is rescaled. We invert with
z_orig = (z_mat - 1) * (0.2604/0.3) + 1 so coords land back on the original MRI grid.

aug_1 == the original (identity) scan, so only aug_1 is exported per patient.

Outputs (written into the model results dir):
    length_test.npy, recon_Y_test_true.npy, recon_Y_test_pred.npy  (400, 2, 3)
    test_index_map.csv ; pred_landmarks_csv/<PAT>.csv ; true_landmarks_csv/<PAT>.csv
"""

import csv
import os
import sys

import numpy as np

import common.MyDataset as MyDataset

AUG_NUM = 50
LANDMARK_NAMES = ["LLSCC ant", "LLSCC post", "RLSCC ant", "RLSCC post"]

# Modality presets. cut_layers order: [[row_d, row_a], [col_d, col_a], [slice_d, slice_a]].
# z_rescale: None, or the ratio r so that z_orig = (z_mat - 1) * r + 1 (maps the isotropic .mat
# grid back to the original anisotropic grid; for MR r = in-plane / z-spacing = 0.2604/0.3).
CONFIG = {
    "ct": {
        "res_dir": ("/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models/cropped/63x50x61/"
                    "noises_s1_test_dis/two_landmarks/straight_model/turner-results/training-process/"
                    "19Nov2025-13:17:24-trainID-33.3"),
        "src_crop_dir": "/data/gpfs/projects/punim1836/Data/cropped/100x100x100/noises_s1_test_dis",
        "length_prefix": "cropped_length",
        "crop_tag": "100x100x100",
        "cut_layers": np.array([[20, 17], [27, 23], [19, 20]]),  # -> 63 x 50 x 61
        "col_size": 50,
        "voxel_mm": 0.15,
        "z_rescale": None,
    },
    "mr": {
        "res_dir": ("/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models_MR/cropped_MR/49x42x42/"
                    "noises_s1_test_dis/two_landmarks/straight_model/learning_rate/0.0001/"
                    "19Sep2024-13:24:39-trainID-8.2"),
        "src_crop_dir": "/data/gpfs/projects/punim1836/Data/cropped_MR/100x100x100/noises_s1_test_dis",
        "length_prefix": "cropped_MR_length",
        "crop_tag": "100x100x100",
        "cut_layers": np.array([[21, 30], [38, 20], [27, 31]]),  # -> 49 x 42 x 42
        "col_size": 42,
        "voxel_mm": 0.2604,
        "z_rescale": 0.2604 / 0.3,
    },
}

# test split patient ids -> names (get_test_pat_splits(): test_pats_id = [8, 7, 1, 15])
TEST_PAT_IDS = MyDataset.get_test_pat_splits()[2]
TEST_PAT_NAMES = [MyDataset.get_pat_names()[i] for i in TEST_PAT_IDS]


def build_length_test(cfg):
    """Load source crop length, apply the cut_layers adjustment (as load_dataset_crop),
    and return the test-split subset (400, 2, 3)."""
    cut = cfg["cut_layers"]
    length = np.load(f"{cfg['src_crop_dir']}/{cfg['length_prefix']}_{cfg['crop_tag']}.npy").astype("float32")
    n = length.shape[0]
    # y/length order is [x=col, y=row, z=slice]
    length[range(0, n, 2)] += [cut[1, 0], cut[0, 0], cut[2, 0]]  # left  += [col_d, row_d, slice_d]
    length[range(1, n, 2)] += [cut[1, 1], cut[0, 0], cut[2, 0]]  # right += [col_a, row_d, slice_d]

    test_idx = MyDataset.get_data_splits(MyDataset.get_test_pat_splits(), split=True, aug_num=AUG_NUM)[2]
    return length[test_idx], test_idx


def reconstruct(y_cropped, length_test, cfg):
    """Un-flip the right (odd) ear on the column axis, add the crop offset, and (MR only)
    rescale z back to the original-scan grid. Returns full-volume voxel coords."""
    p = np.copy(y_cropped).astype("float64")
    for k in range(p.shape[0] // 2):
        right = 2 * k + 1
        p[right, :, 0] = (cfg["col_size"] + 1) - p[right, :, 0]
    p = p + length_test
    if cfg["z_rescale"] is not None:
        p[:, :, 2] = (p[:, :, 2] - 1.0) * cfg["z_rescale"] + 1.0
    return p


def build_index_map(n_rows):
    """Row -> (patient, aug_id, ear). Order: TEST_PAT_NAMES; per patient 50 augs x 2 ears
    interleaved (aug1-left, aug1-right, aug2-left, ...)."""
    rows = []
    per_pat = AUG_NUM * 2
    for r in range(n_rows):
        pat = TEST_PAT_NAMES[r // per_pat]
        within = r % per_pat
        rows.append((r, pat, within // 2 + 1, "left" if within % 2 == 0 else "right"))
    return rows


def write_aug1_csvs(recon, out_dir, tag):
    """Write one CSV per test patient with the 4 landmarks of aug_1 (= original scan).
    recon: (400, 2, 3) full-volume voxel coords. aug_1 = first 2 rows of each 100-row block
    (left ear = ant,post ; right ear = ant,post). Columns: name,x,y,z (1-based voxel, not rounded)."""
    os.makedirs(out_dir, exist_ok=True)
    per_pat = AUG_NUM * 2
    for bi, pat in enumerate(TEST_PAT_NAMES):
        base = bi * per_pat  # aug_1 left = base, right = base + 1
        left, right = recon[base], recon[base + 1]  # each (2, 3): ant, post
        pts = [left[0], left[1], right[0], right[1]]
        with open(f"{out_dir}/{pat}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "x", "y", "z"])
            for name, p in zip(LANDMARK_NAMES, pts):
                w.writerow([name, float(p[0]), float(p[1]), float(p[2])])
    print(f"  [{tag}] wrote {len(TEST_PAT_NAMES)} CSVs to {out_dir}")


def main():
    modality = (sys.argv[1].lower() if len(sys.argv) > 1 else "ct")
    if modality not in CONFIG:
        raise SystemExit(f"unknown modality '{modality}'; choose one of {list(CONFIG)}")
    cfg = CONFIG[modality]
    res_dir = cfg["res_dir"]
    print(f"Modality: {modality}  (voxel {cfg['voxel_mm']} mm, z_rescale {cfg['z_rescale']})")

    length_test, test_idx = build_length_test(cfg)
    print("Test patients:", TEST_PAT_NAMES, "ids:", list(TEST_PAT_IDS))
    print("test_idx count:", len(test_idx), "first/last:", test_idx[:4], test_idx[-4:])

    y_true = np.load(f"{res_dir}/Y_test_true.npy")
    y_pred = np.load(f"{res_dir}/final_Y_test_pred.npy")
    assert y_true.shape == y_pred.shape == length_test.shape, \
        f"shape mismatch: true {y_true.shape}, pred {y_pred.shape}, length {length_test.shape}"

    recon_true = reconstruct(y_true, length_test, cfg).astype("float32")
    recon_pred = reconstruct(y_pred, length_test, cfg).astype("float32")

    # ---- save ----
    np.save(f"{res_dir}/length_test.npy", length_test.astype("float32"))
    np.save(f"{res_dir}/recon_Y_test_true.npy", recon_true)
    np.save(f"{res_dir}/recon_Y_test_pred.npy", recon_pred)

    with open(f"{res_dir}/test_index_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "patient", "aug_id", "ear"])
        w.writerows(build_index_map(recon_true.shape[0]))

    # ---- per-patient aug_1 (original-scan) landmark CSVs ----
    write_aug1_csvs(recon_pred, f"{res_dir}/pred_landmarks_csv", "pred")
    write_aug1_csvs(recon_true, f"{res_dir}/true_landmarks_csv", "true")

    print("Saved:")
    for name in ["length_test.npy", "recon_Y_test_true.npy", "recon_Y_test_pred.npy", "test_index_map.csv"]:
        print(f"  {res_dir}/{name}")

    # ---- self-check ----
    # NOTE: with z_rescale the per-axis mm scale differs slightly in z; this is a coarse sanity figure.
    err = np.linalg.norm((recon_pred - recon_true) * cfg["voxel_mm"], axis=2)  # (400, 2), mm
    print(f"\nmean err: {err.mean():.4f} mm  (ant {err[:, 0].mean():.4f}, post {err[:, 1].mean():.4f})")
    even = recon_true[range(0, recon_true.shape[0], 2)].reshape(-1, 3)  # left ears
    odd = recon_true[range(1, recon_true.shape[0], 2)].reshape(-1, 3)   # right ears
    print(f"LEFT  ear x [min,med,max]: {even[:,0].min():.1f}, {np.median(even[:,0]):.1f}, {even[:,0].max():.1f}")
    print(f"RIGHT ear x [min,med,max]: {odd[:,0].min():.1f}, {np.median(odd[:,0]):.1f}, {odd[:,0].max():.1f}")
    print(f"all coords  min: {recon_true.reshape(-1,3).min(0).round(1)}  max: {recon_true.reshape(-1,3).max(0).round(1)}")


if __name__ == "__main__":
    main()
