"""
Reconstruct the held-out test-set predicted landmarks from cropped (63x50x61) voxel
space back to full-volume voxel coordinates, and persist them.

Target model: down_net (straight_model) FC, train_id 33.3.
Test set: final_Y_test_pred.npy / Y_test_true.npy, shape (400, 2, 3)
          = patients [GM, GE, AZ, LP] x 2 ears x 50 augmentations.

The model's eval path (load_dataset_manager) never saves a per-instance crop offset,
so we rebuild it here from the source crop dataset's length array, apply the same
cut_layers adjustment as load_dataset_crop, take the test split, and reconstruct
(un-flip the right ear on the column axis, then add length).

Outputs (written into the trainID-33.3 results dir):
    length_test.npy        (400, 2, 3)
    recon_Y_test_true.npy  (400, 2, 3)  full-volume voxel coords
    recon_Y_test_pred.npy  (400, 2, 3)  full-volume voxel coords
    test_index_map.csv     row,patient,aug_id,ear
"""

import csv
import numpy as np

import common.MyDataset as MyDataset

# ---- 33.3 constants (mirrors start_training.py train_id 33.3 / evaluate_dataset.py style) ----
RES_DIR = ("/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models/cropped/63x50x61/"
           "noises_s1_test_dis/two_landmarks/straight_model/turner-results/training-process/"
           "19Nov2025-13:17:24-trainID-33.3")
SRC_CROP_DIR = "/data/gpfs/projects/punim1836/Data/cropped/100x100x100/noises_s1_test_dis"

CROP_TAG = "100x100x100"
# cut_layers order: [[row_d, row_a], [col_d, col_a], [slice_d, slice_a]]
CUT_LAYERS = np.array([[20, 17], [27, 23], [19, 20]])
# cropped model space after cut: 100-20-17=63 (row), 100-27-23=50 (col), 100-19-20=61 (slice)
(ROW_SIZE, COLUMN_SIZE, SLICE_SIZE) = (63, 50, 61)

AUG_NUM = 50
# test split patient ids -> names (get_test_pat_splits(): test_pats_id = [8, 7, 1, 15])
TEST_PAT_IDS = MyDataset.get_test_pat_splits()[2]
PAT_NAMES_ALL = MyDataset.get_pat_names()
TEST_PAT_NAMES = [PAT_NAMES_ALL[i] for i in TEST_PAT_IDS]


def build_length_test():
    """Load source crop length, apply the cut_layers adjustment (as load_dataset_crop),
    and return the test-split subset (400, 2, 3)."""
    length = np.load(f"{SRC_CROP_DIR}/cropped_length_{CROP_TAG}.npy").astype("float32")
    n = length.shape[0]
    # y/length order is [x=col, y=row, z=slice]
    length[range(0, n, 2)] += [CUT_LAYERS[1, 0], CUT_LAYERS[0, 0], CUT_LAYERS[2, 0]]  # left  += [col_d, row_d, slice_d]
    length[range(1, n, 2)] += [CUT_LAYERS[1, 1], CUT_LAYERS[0, 0], CUT_LAYERS[2, 0]]  # right += [col_a, row_d, slice_d]

    test_idx = MyDataset.get_data_splits(MyDataset.get_test_pat_splits(), split=True, aug_num=AUG_NUM)[2]
    return length[test_idx], test_idx


def reconstruct(y_cropped, length_test):
    """Un-flip the right (odd) ear on the column axis, then add the crop offset.
    y_cropped: (N, 2, 3) in cropped 63x50x61 space; returns full-volume voxel coords."""
    p = np.copy(y_cropped).astype("float64")
    for k in range(p.shape[0] // 2):
        right = 2 * k + 1
        p[right, :, 0] = (COLUMN_SIZE + 1) - p[right, :, 0]
    return p + length_test


def build_index_map(n_rows):
    """Row -> (patient, aug_id, ear). Order: TEST_PAT_NAMES; per patient 50 augs x 2 ears
    interleaved (aug1-left, aug1-right, aug2-left, ...)."""
    rows = []
    per_pat = AUG_NUM * 2
    for r in range(n_rows):
        pat = TEST_PAT_NAMES[r // per_pat]
        within = r % per_pat
        aug_id = within // 2 + 1
        ear = "left" if within % 2 == 0 else "right"
        rows.append((r, pat, aug_id, ear))
    return rows


LANDMARK_NAMES = ["LLSCC ant", "LLSCC post", "RLSCC ant", "RLSCC post"]


def write_aug1_csvs(recon, out_dir, tag):
    """Write one CSV per test patient with the 4 landmarks of aug_1 (= original scan).
    recon: (400, 2, 3) full-volume voxel coords. aug_1 = first 2 rows of each 100-row block
    (left ear = ant,post ; right ear = ant,post). Columns: name,x,y,z (1-based voxel, not rounded)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    per_pat = AUG_NUM * 2
    written = []
    for bi, pat in enumerate(TEST_PAT_NAMES):
        base = bi * per_pat  # aug_1 left = base, right = base + 1
        left, right = recon[base], recon[base + 1]  # each (2, 3): ant, post
        pts = [left[0], left[1], right[0], right[1]]
        path = f"{out_dir}/{pat}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "x", "y", "z"])
            for name, p in zip(LANDMARK_NAMES, pts):
                w.writerow([name, float(p[0]), float(p[1]), float(p[2])])
        written.append(path)
    print(f"  [{tag}] wrote {len(written)} CSVs to {out_dir}")
    return written


def main():
    length_test, test_idx = build_length_test()
    print("Test patients:", TEST_PAT_NAMES, "ids:", list(TEST_PAT_IDS))
    print("test_idx count:", len(test_idx), "first/last:", test_idx[:4], test_idx[-4:])

    y_true = np.load(f"{RES_DIR}/Y_test_true.npy")
    y_pred = np.load(f"{RES_DIR}/final_Y_test_pred.npy")
    assert y_true.shape == y_pred.shape == length_test.shape, \
        f"shape mismatch: true {y_true.shape}, pred {y_pred.shape}, length {length_test.shape}"

    recon_true = reconstruct(y_true, length_test).astype("float32")
    recon_pred = reconstruct(y_pred, length_test).astype("float32")

    # ---- save ----
    np.save(f"{RES_DIR}/length_test.npy", length_test.astype("float32"))
    np.save(f"{RES_DIR}/recon_Y_test_true.npy", recon_true)
    np.save(f"{RES_DIR}/recon_Y_test_pred.npy", recon_pred)

    index_map = build_index_map(recon_true.shape[0])
    with open(f"{RES_DIR}/test_index_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "patient", "aug_id", "ear"])
        w.writerows(index_map)

    # ---- per-patient aug_1 (original-scan) landmark CSVs ----
    write_aug1_csvs(recon_pred, f"{RES_DIR}/pred_landmarks_csv", "pred")
    write_aug1_csvs(recon_true, f"{RES_DIR}/true_landmarks_csv", "true")

    print("Saved:")
    for name in ["length_test.npy", "recon_Y_test_true.npy", "recon_Y_test_pred.npy", "test_index_map.csv"]:
        print(f"  {RES_DIR}/{name}")

    # ---- self-check ----
    err = np.linalg.norm((recon_pred - recon_true) * 0.15, axis=2)  # (400, 2), mm @ 0.15
    print(f"\nmean err: {err.mean():.4f} mm  (ant {err[:, 0].mean():.4f}, post {err[:, 1].mean():.4f})")
    even = recon_true[range(0, recon_true.shape[0], 2)].reshape(-1, 3)  # left ears
    odd = recon_true[range(1, recon_true.shape[0], 2)].reshape(-1, 3)   # right ears
    print(f"LEFT  ear x [min,med,max]: {even[:,0].min():.1f}, {np.median(even[:,0]):.1f}, {even[:,0].max():.1f}")
    print(f"RIGHT ear x [min,med,max]: {odd[:,0].min():.1f}, {np.median(odd[:,0]):.1f}, {odd[:,0].max():.1f}")
    print(f"all coords  min: {recon_true.reshape(-1,3).min(0).round(1)}  max: {recon_true.reshape(-1,3).max(0).round(1)}")


if __name__ == "__main__":
    main()
