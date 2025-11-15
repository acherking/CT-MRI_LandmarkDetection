import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union


def convert_roi_addition_table(
    xlsx_path: str,
    out_dir: str,
    round_coords: bool = False,
    strip_suffix: str = " Pre",
) -> Dict[str, str]:
    """
    Convert ROI_addition_CT_Pre_14_Nov2025.xlsx into per-patient landmarks_*.csv.

    Expected structure:

        Case number | Imaging modality | ROI | (blank) | (blank) | (blank)
                     (next row)          X   |   Y     |   Z
        CD Pre      | CT               | LLSCC ant | 800 | 402 | 244
                     |                  LLSCC post| 786 | 429 | 237
                     |                  ...

    Behavior:
        - Uses "Case number" as patient ID, strips `strip_suffix` (e.g. " Pre") for filenames.
        - Ignores "Imaging modality".
        - Uses column 3 ("ROI") as landmark name.
        - Uses columns 4,5,6 (X,Y,Z) as coordinates.
        - Writes one CSV per patient: <patient>_landmarks.csv
          with columns: name,x,y,z
    """

    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read as-is; top row is header, second row may be X/Y/Z labels under the ROI block.
    df = pd.read_excel(xlsx_path)

    # Column layout (by position) based on the example:
    case_col = df.columns[0]      # "Case number"
    modality_col = df.columns[1]  # "Imaging modality" (unused)
    roi_col = df.columns[2]       # "ROI"
    x_col = df.columns[3]
    y_col = df.columns[4]
    z_col = df.columns[5]

    # Drop any "header" row that has ROI == "ROI" or X == "X"
    mask_header_like = (
        (df[roi_col].astype(str).str.strip() == "ROI")
        | (df[x_col].astype(str).str.strip() == "X")
    )
    df = df[~mask_header_like].copy()

    # Forward-fill case numbers downwards (since only first row of each block has it)
    df[case_col] = df[case_col].ffill()

    # Build per-patient records
    per_patient: Dict[str, List[Dict[str, Union[str, float, int]]]] = {}

    for _, row in df.iterrows():
        raw_case = str(row[case_col]).strip()
        if not raw_case or raw_case.lower() == "nan":
            continue

        # Strip suffix like " Pre" if present
        if strip_suffix and raw_case.endswith(strip_suffix):
            patient = raw_case[: -len(strip_suffix)].strip()
        else:
            patient = raw_case

        roi_name = str(row[roi_col]).strip()
        if not roi_name or roi_name.lower() == "nan":
            continue

        x = row[x_col]
        y = row[y_col]
        z = row[z_col]
        if any(pd.isna(v) for v in (x, y, z)):
            # skip rows without complete coordinates
            continue

        x_f, y_f, z_f = float(x), float(y), float(z)
        if round_coords:
            x_out = int(round(x_f))
            y_out = int(round(y_f))
            z_out = int(round(z_f))
        else:
            x_out, y_out, z_out = x_f, y_f, z_f

        per_patient.setdefault(patient, []).append(
            {"name": roi_name, "x": x_out, "y": y_out, "z": z_out}
        )

    written: Dict[str, str] = {}
    for patient, recs in per_patient.items():
        if not recs:
            continue
        out_path = out_dir / f"{patient}_landmarks.csv"
        pd.DataFrame(recs, columns=["name", "x", "y", "z"]).to_csv(out_path, index=False)
        written[patient] = str(out_path)

    return written


if __name__ == "__main__":
    xlsx_path = "/data/gpfs/projects/punim1836/Data/raw/ROI/more_ct_14/ROI_addition_CT_Pre_14_Nov2025.xlsx"
    out_dir = "/data/gpfs/projects/punim1836/Data/raw/landmarks/more_CT_Pre_14"

    written = convert_roi_addition_table(
        xlsx_path=xlsx_path,
        out_dir=out_dir,
        round_coords=False,      # set True if you want integer voxel indices
        strip_suffix=" Pre",     # remove " Pre" from case number
    )

    print("Written landmark CSVs:")
    for p, path in written.items():
        print(f"  {p}: {path}")
