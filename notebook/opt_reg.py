"""
SimpleITK-based CT/MRI rigid registration functions.
Extracted from opt_reg.ipynb.

Three registration modes:
  1. rigid_ct_fixed_mri_moving        - basic rigid registration with iso resampling
  2. refine_rigid_mi_jupyter          - rigid MI registration with before/after metrics
  3. micro_refine_rigid_mi_none_sampling - micro-refinement using full-mask (NONE) sampling
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def to_float(img: sitk.Image) -> sitk.Image:
    """Cast image to float32 if not already."""
    return sitk.Cast(img, sitk.sitkFloat32) if img.GetPixelID() != sitk.sitkFloat32 else img


def resample_iso(img: sitk.Image, iso: float = 0.6,
                 interp=sitk.sitkLinear) -> sitk.Image:
    """Resample image to isotropic spacing."""
    img = to_float(img)
    sp = np.array(img.GetSpacing(), float)
    sz = np.array(img.GetSize(), int)
    new_spacing = np.array([iso, iso, iso], float)
    new_size = np.maximum(1, np.round(sz * (sp / new_spacing)).astype(int)).tolist()
    return sitk.Resample(
        img, new_size, sitk.Transform(), interp,
        img.GetOrigin(), new_spacing.tolist(), img.GetDirection(),
        0.0, img.GetPixelIDValue()
    )


def pick_pyramid(sz, sp):
    """Choose pyramid shrink/smooth levels based on Z extent."""
    phys_z = sz[2] * sp[2]
    if sz[2] < 40 or phys_z < 30:
        return [3, 2, 1], [1.5, 1.0, 0.0]
    elif sz[2] < 80 or phys_z < 60:
        return [4, 2, 1], [2.0, 1.0, 0.0]
    else:
        return [6, 3, 1], [3.0, 1.5, 0.0]


def print_props(name: str, img: sitk.Image):
    print(f"{name}:")
    print("  Direction:", img.GetDirection())
    print("  Spacing  :", img.GetSpacing())
    print("  Origin   :", img.GetOrigin())
    print("  Size     :", img.GetSize())


def resample_to_ref(moving: sitk.Image, reference: sitk.Image,
                    tx: sitk.Transform, interp=sitk.sitkLinear,
                    default_value: float = 0.0) -> sitk.Image:
    return sitk.Resample(moving, reference, tx, interp, default_value, moving.GetPixelID())


def resample_mask_to_ref(moving_mask: sitk.Image, reference: sitk.Image,
                         tx: sitk.Transform) -> sitk.Image:
    out = sitk.Resample(moving_mask, reference, tx,
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


# ---------------------------------------------------------------------------
# Array-based metrics
# ---------------------------------------------------------------------------

def _masked_arrays(fixed: sitk.Image, moved: sitk.Image,
                   mask: Optional[sitk.Image] = None,
                   max_voxels: int = 500_000):
    """Return 1D float arrays (fixed, moved), optionally masked & downsampled."""
    f = sitk.GetArrayFromImage(fixed).astype(np.float32).ravel()
    m = sitk.GetArrayFromImage(moved).astype(np.float32).ravel()
    if mask is not None:
        mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8).ravel() > 0
        f = f[mask_arr]
        m = m[mask_arr]
    n = f.size
    if n > max_voxels:
        rs = np.random.RandomState(0)
        idx = rs.choice(n, size=max_voxels, replace=False)
        f = f[idx]; m = m[idx]
    return f, m


def mse(f: np.ndarray, g: np.ndarray) -> float:
    return float(np.mean((f - g) ** 2))


def ncc(f: np.ndarray, g: np.ndarray, eps: float = 1e-8) -> float:
    f = f - f.mean(); g = g - g.mean()
    num = float(np.dot(f, g))
    den = float(np.linalg.norm(f) * np.linalg.norm(g)) + eps
    return num / den


def gradient_image(img: sitk.Image, sigma: float = 1.0) -> sitk.Image:
    gm = sitk.GradientMagnitudeRecursiveGaussian(img, sigma)
    arr = sitk.GetArrayFromImage(gm).astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return sitk.GetImageFromArray(arr)


# ---------------------------------------------------------------------------
# 1. rigid_ct_fixed_mri_moving
#    Basic rigid registration: CT fixed, MRI moving.
#    Optionally resamples both to isotropic spacing for registration,
#    then applies the transform back to full-res MRI.
# ---------------------------------------------------------------------------

def rigid_ct_fixed_mri_moving(
    ct_path: str,
    mri_path: str,
    out_moved: str = "mri_in_ct_space_rigid.nii.gz",
    out_tfm: str = "rigid_ct_from_mri.tfm",
    use_iso: bool = True,
    iso: float = 0.6,
    init_mode: str = "GEOMETRY",       # or "MOMENTS"
    bins: int = 50,
    samp: float = 0.20,
    iters: int = 300,
    lr: float = 2.0,
    min_step: float = 1e-3,
    relax: float = 0.5,
    fixed_mask_path: Optional[str] = None,
    moving_mask_path: Optional[str] = None,
):
    """
    Rigid CT-fixed / MRI-moving registration using Mattes MI.

    Parameters
    ----------
    ct_path, mri_path     : paths to CT (fixed) and MRI (moving) images
    out_moved             : output path for warped MRI in CT space
    out_tfm               : output path for the rigid transform (.tfm)
    use_iso               : resample both images to isotropic spacing before registration
    iso                   : isotropic resolution in mm (used when use_iso=True)
    init_mode             : "GEOMETRY" or "MOMENTS" for centred initializer
    bins                  : number of histogram bins for Mattes MI
    samp                  : random sampling fraction for MI (0–1)
    iters                 : maximum optimizer iterations
    lr                    : optimizer learning rate
    min_step              : optimizer minimum step
    relax                 : optimizer relaxation factor
    fixed_mask_path       : optional UInt8 mask image for CT (0/1)
    moving_mask_path      : optional UInt8 mask image for MRI (0/1)

    Returns
    -------
    (final_tx, info_dict, moved_image)
    """
    ct  = to_float(sitk.ReadImage(ct_path))
    mri = to_float(sitk.ReadImage(mri_path))

    fixed_mask  = sitk.ReadImage(fixed_mask_path)  if fixed_mask_path  else None
    moving_mask = sitk.ReadImage(moving_mask_path) if moving_mask_path else None

    if use_iso:
        ct_reg   = resample_iso(ct,  iso=iso, interp=sitk.sitkLinear)
        mri_reg  = resample_iso(mri, iso=iso, interp=sitk.sitkLinear)
        fmask_reg = resample_iso(fixed_mask,  iso=iso, interp=sitk.sitkNearestNeighbor) if fixed_mask  else None
        mmask_reg = resample_iso(moving_mask, iso=iso, interp=sitk.sitkNearestNeighbor) if moving_mask else None
    else:
        ct_reg, mri_reg = ct, mri
        fmask_reg, mmask_reg = fixed_mask, moving_mask

    shrink, smooth = pick_pyramid(ct_reg.GetSize(), ct_reg.GetSpacing())

    init_enum = (sitk.CenteredTransformInitializerFilter.GEOMETRY
                 if init_mode.upper() == "GEOMETRY"
                 else sitk.CenteredTransformInitializerFilter.MOMENTS)
    init_tx = sitk.CenteredTransformInitializer(
        ct_reg, mri_reg, sitk.Euler3DTransform(), init_enum
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(bins))
    if fmask_reg: R.SetMetricFixedMask(fmask_reg)
    if mmask_reg: R.SetMetricMovingMask(mmask_reg)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(float(samp))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(lr),
        minStep=float(min_step),
        numberOfIterations=int(iters),
        relaxationFactor=float(relax),
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrink)
    R.SetSmoothingSigmasPerLevel(smooth)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init_tx, inPlace=False)

    final_tx = R.Execute(ct_reg, mri_reg)

    # Apply transform in physical space back to full-res MRI
    moved_full = sitk.Resample(mri, ct, final_tx, sitk.sitkLinear, 0.0, mri.GetPixelID())

    sitk.WriteImage(moved_full, out_moved)
    sitk.WriteTransform(final_tx, out_tfm)

    info = {
        "final_metric_value": R.GetMetricValue(),
        "stop": R.GetOptimizerStopConditionDescription(),
        "iterations": R.GetOptimizerIteration(),
        "pyramid_shrink": shrink,
        "pyramid_smooth_mm": smooth,
        "used_iso": use_iso,
        "iso_mm": iso,
        "init_mode": init_mode,
        "ct_reg_size": tuple(ct_reg.GetSize()),
        "mri_reg_size": tuple(mri_reg.GetSize()),
    }
    return final_tx, info, moved_full


# ---------------------------------------------------------------------------
# 2. refine_rigid_mi_jupyter
#    Rigid Mattes-MI registration with detailed before/after evaluation:
#    intensity MSE, NCC, gradient NCC, and optional mask Dice.
# ---------------------------------------------------------------------------

def configure_mi_rigid(
    R: sitk.ImageRegistrationMethod,
    histogram_bins: int = 50,
    sampling_perc: float = 0.30,
    shrink=(8, 4, 2, 1),
    smooth_mm=(3.0, 2.0, 1.0, 0.0),
    use_line_search: bool = True,
):
    """Configure Mattes MI metric and optimizer on an ImageRegistrationMethod."""
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(histogram_bins))
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(float(sampling_perc))
    R.SetMetricSamplingSeed(42)
    R.SetInterpolator(sitk.sitkLinear)

    if use_line_search:
        R.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10,
        )
    else:
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=4.0,
            minStep=1e-5,
            numberOfIterations=400,
            relaxationFactor=0.5,
        )
        R.SetOptimizerScalesFromPhysicalShift()

    R.SetShrinkFactorsPerLevel(list(shrink))
    R.SetSmoothingSigmasPerLevel(list(smooth_mm))
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


def evaluate_mi_cost(
    fixed: sitk.Image,
    moving: sitk.Image,
    tx: sitk.Transform,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
    histogram_bins: int = 64,
    sampling_perc: float = 0.40,
    shrink=(4, 2, 1),
    smooth_mm=(2.0, 1.0, 0.0),
) -> float:
    """Evaluate Mattes MI cost without optimizing. Lower cost = better alignment."""
    R = sitk.ImageRegistrationMethod()
    configure_mi_rigid(R, histogram_bins, sampling_perc, shrink, smooth_mm)
    if fixed_mask  is not None: R.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None: R.SetMetricMovingMask(moving_mask)
    R.SetInitialTransform(tx, inPlace=False)
    return float(R.MetricEvaluate(fixed, moving))


def refine_rigid_mi_jupyter(
    ct_path: str,
    mri_path: str,
    out_warped_path: str = "mri_in_ct_space_rigid.nii.gz",
    out_transform_path: str = "rigid_ct_from_mri.tfm",
    init_mode: str = "GEOMETRY",            # or "MOMENTS"
    histogram_bins: int = 64,
    sampling_perc: float = 0.40,
    fixed_mask_path: Optional[str] = None,
    moving_mask_path: Optional[str] = None,
    report_gradient_sigma: float = 1.0,
):
    """
    Rigid CT-fixed / MRI-moving registration with comprehensive before/after reporting.

    Reports: Mattes MI cost, intensity MSE, intensity NCC, gradient NCC, mask Dice.

    Parameters
    ----------
    ct_path, mri_path         : paths to CT (fixed) and MRI (moving) images
    out_warped_path           : output path for warped MRI
    out_transform_path        : output path for rigid transform (.tfm)
    init_mode                 : "GEOMETRY" or "MOMENTS"
    histogram_bins            : Mattes MI histogram bins
    sampling_perc             : random sampling fraction (0–1)
    fixed_mask_path           : optional UInt8 mask in CT space
    moving_mask_path          : optional UInt8 mask in MRI space
    report_gradient_sigma     : Gaussian sigma (mm) for gradient NCC evaluation

    Returns
    -------
    dict with keys: final_transform, mri_warped, before, after
    """
    ct  = sitk.ReadImage(ct_path,  sitk.sitkFloat32)
    mri = sitk.ReadImage(mri_path, sitk.sitkFloat32)

    print("=== Image properties ===")
    print_props("CT ", ct)
    print_props("MRI", mri)

    fixed_mask  = sitk.ReadImage(fixed_mask_path,  sitk.sitkUInt8) if fixed_mask_path  else None
    moving_mask = sitk.ReadImage(moving_mask_path, sitk.sitkUInt8) if moving_mask_path else None

    shrink    = (4, 2, 1)
    smooth_mm = (2.0, 1.0, 0.0)

    init_enum = (sitk.CenteredTransformInitializerFilter.GEOMETRY
                 if init_mode.upper() == "GEOMETRY"
                 else sitk.CenteredTransformInitializerFilter.MOMENTS)
    init_tx = sitk.CenteredTransformInitializer(
        ct, mri, sitk.VersorRigid3DTransform(), init_enum
    )

    # --- Evaluate BEFORE ---
    pre_cost   = evaluate_mi_cost(ct, mri, init_tx, fixed_mask, moving_mask,
                                  histogram_bins, sampling_perc, shrink, smooth_mm)
    pre_mi_est = -pre_cost
    mri_pre    = resample_to_ref(mri, ct, init_tx)

    eval_mask = None
    if fixed_mask is not None and moving_mask is not None:
        moving_mask_pre = resample_mask_to_ref(moving_mask, ct, init_tx)
        eval_mask = sitk.Cast(sitk.And(fixed_mask > 0, moving_mask_pre > 0), sitk.sitkUInt8)
    elif fixed_mask is not None:
        eval_mask = fixed_mask

    f_arr, pre_arr = _masked_arrays(ct, mri_pre, eval_mask)
    pre_mse   = mse(f_arr, pre_arr)
    pre_ncc_v = ncc(f_arr, pre_arr)

    ct_g         = gradient_image(ct, sigma=report_gradient_sigma)
    mri_pre_g    = gradient_image(mri_pre, sigma=report_gradient_sigma)
    f_g, pre_g   = _masked_arrays(ct_g, mri_pre_g, eval_mask)
    pre_g_ncc    = ncc(f_g, pre_g)

    pre_dice = None
    if fixed_mask is not None and moving_mask is not None:
        inter = sitk.LabelOverlapMeasuresImageFilter()
        inter.Execute(fixed_mask > 0, moving_mask_pre > 0)
        pre_dice = inter.GetDiceCoefficient()

    print("\n=== BEFORE refinement ===")
    print(f"Mattes MI cost (↓):  {pre_cost: .6f}   MI est (↑): {pre_mi_est: .6f}")
    print(f"Intensity MSE  (↓):  {pre_mse: .6f}")
    print(f"Intensity NCC  (↑):  {pre_ncc_v: .6f}")
    print(f"Gradient NCC   (↑):  {pre_g_ncc: .6f}  (σ={report_gradient_sigma})")
    if pre_dice is not None:
        print(f"Mask Dice      (↑):  {pre_dice: .6f}")

    # --- Optimize ---
    R = sitk.ImageRegistrationMethod()
    configure_mi_rigid(
        R, histogram_bins=histogram_bins, sampling_perc=sampling_perc,
        shrink=(8, 4, 2, 1), smooth_mm=(3.0, 2.0, 1.0, 0.0),
        use_line_search=True,
    )
    if fixed_mask  is not None: R.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None: R.SetMetricMovingMask(moving_mask)
    R.SetInitialTransform(init_tx, inPlace=False)
    final_tx = R.Execute(ct, mri)

    # --- Evaluate AFTER ---
    post_cost   = evaluate_mi_cost(ct, mri, final_tx, fixed_mask, moving_mask,
                                   histogram_bins, sampling_perc, shrink, smooth_mm)
    post_mi_est = -post_cost
    mri_post    = resample_to_ref(mri, ct, final_tx)

    eval_mask_post = None
    if fixed_mask is not None and moving_mask is not None:
        moving_mask_post = resample_mask_to_ref(moving_mask, ct, final_tx)
        eval_mask_post = sitk.Cast(sitk.And(fixed_mask > 0, moving_mask_post > 0), sitk.sitkUInt8)
    elif fixed_mask is not None:
        eval_mask_post = fixed_mask

    f_arr2, post_arr = _masked_arrays(ct, mri_post, eval_mask_post)
    post_mse   = mse(f_arr2, post_arr)
    post_ncc_v = ncc(f_arr2, post_arr)

    mri_post_g      = gradient_image(mri_post, sigma=report_gradient_sigma)
    f_g2, post_g    = _masked_arrays(ct_g, mri_post_g, eval_mask_post)
    post_g_ncc      = ncc(f_g2, post_g)

    post_dice = None
    if fixed_mask is not None and moving_mask is not None:
        inter2 = sitk.LabelOverlapMeasuresImageFilter()
        inter2.Execute(fixed_mask > 0, moving_mask_post > 0)
        post_dice = inter2.GetDiceCoefficient()

    print("\n=== AFTER refinement ===")
    print(f"Mattes MI cost (↓):  {post_cost: .6f}   MI est (↑): {post_mi_est: .6f}")
    print(f"Intensity MSE  (↓):  {post_mse: .6f}    Δ = {post_mse - pre_mse:+.6f}")
    print(f"Intensity NCC  (↑):  {post_ncc_v: .6f}    Δ = {post_ncc_v - pre_ncc_v:+.6f}")
    print(f"Gradient NCC   (↑):  {post_g_ncc: .6f}    Δ = {post_g_ncc - pre_g_ncc:+.6f}")
    if post_dice is not None:
        print(f"Mask Dice      (↑):  {post_dice: .6f}    Δ = {post_dice - pre_dice:+.6f}")
    print("\nStop condition:", R.GetOptimizerStopConditionDescription())
    print("Iterations    :", R.GetOptimizerIteration())

    sitk.WriteImage(mri_post, out_warped_path)
    sitk.WriteTransform(final_tx, out_transform_path)
    print(f"\nSaved warped MRI : {out_warped_path}")
    print(f"Saved transform  : {out_transform_path}")

    return {
        "final_transform": final_tx,
        "mri_warped": mri_post,
        "before": dict(mi_cost=pre_cost,  mi=pre_mi_est,  mse=pre_mse,  ncc=pre_ncc_v,  g_ncc=pre_g_ncc,  dice=pre_dice),
        "after":  dict(mi_cost=post_cost, mi=post_mi_est, mse=post_mse, ncc=post_ncc_v, g_ncc=post_g_ncc, dice=post_dice),
    }


# ---------------------------------------------------------------------------
# 3. micro_refine_rigid_mi_none_sampling
#    Micro-refinement using full-mask (NONE) sampling — stable for small ROIs,
#    suitable for already pre-aligned images.
# ---------------------------------------------------------------------------

def _mi_value(
    fixed: sitk.Image,
    moving: sitk.Image,
    tx: sitk.Transform,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
    bins: int = 32,
) -> float:
    """Evaluate Mattes MI (higher = better) with NONE sampling."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    R.SetMetricSamplingStrategy(R.NONE)
    if fixed_mask  is not None: R.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None: R.SetMetricMovingMask(moving_mask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=False)
    return -float(R.MetricEvaluate(fixed, moving))


def micro_refine_rigid_mi_none_sampling(
    ct_path: str,
    mri_path: str,
    fixed_mask_path: Optional[str] = None,
    moving_mask_path: Optional[str] = None,
    init_mode: str = "GEOMETRY",            # or "MOMENTS"
    bins: int = 32,
    threads: int = 8,
    out_warped_path: Optional[str] = "mri_refined_in_ct_space.nii.gz",
    out_transform_path: Optional[str] = "refine_micro.tfm",
):
    """
    Micro-refinement of a nearly-aligned CT/MRI pair using Mattes MI with
    NONE sampling (all masked voxels) — maximally stable for small corrections.

    Parameters
    ----------
    ct_path, mri_path         : paths to CT (fixed) and MRI (moving) images
    fixed_mask_path           : optional UInt8 mask in CT space
    moving_mask_path          : optional UInt8 mask in MRI space
    init_mode                 : "GEOMETRY" or "MOMENTS"
    bins                      : Mattes MI histogram bins (32 is enough for micro-refine)
    threads                   : number of CPU threads for SimpleITK
    out_warped_path           : output path for refined MRI (None to skip)
    out_transform_path        : output path for transform (None to skip)

    Returns
    -------
    dict with keys: mi_before, mi_after, transform, stop, iters
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))

    ct  = sitk.ReadImage(ct_path,  sitk.sitkFloat32)
    mri = sitk.ReadImage(mri_path, sitk.sitkFloat32)

    fixed_mask  = sitk.ReadImage(fixed_mask_path,  sitk.sitkUInt8) if fixed_mask_path  else None
    moving_mask = sitk.ReadImage(moving_mask_path, sitk.sitkUInt8) if moving_mask_path else None

    init_enum = (sitk.CenteredTransformInitializerFilter.GEOMETRY
                 if init_mode.upper() == "GEOMETRY"
                 else sitk.CenteredTransformInitializerFilter.MOMENTS)
    try:
        base_tx = sitk.VersorRigid3DTransform()
    except Exception:
        base_tx = sitk.Euler3DTransform()

    init_tx = sitk.CenteredTransformInitializer(ct, mri, base_tx, init_enum)

    mi_before = _mi_value(ct, mri, init_tx, fixed_mask, moving_mask, bins=bins)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    R.SetMetricSamplingStrategy(R.NONE)
    if fixed_mask  is not None: R.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None: R.SetMetricMovingMask(moving_mask)
    R.SetInterpolator(sitk.sitkLinear)

    if hasattr(R, "SetOptimizerAsGradientDescentLineSearch"):
        R.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.3,
            numberOfIterations=50,
            convergenceMinimumValue=5e-6,
            convergenceWindowSize=8,
        )
    else:
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.2,
            minStep=1e-6,
            numberOfIterations=50,
            relaxationFactor=0.7,
        )
        R.SetOptimizerScalesFromPhysicalShift()

    R.SetShrinkFactorsPerLevel([1])
    R.SetSmoothingSigmasPerLevel([0.0])
    R.SetInitialTransform(init_tx, inPlace=False)
    final_tx = R.Execute(ct, mri)

    mi_after = _mi_value(ct, mri, final_tx, fixed_mask, moving_mask, bins=bins)

    print("=== Micro-refine (NONE sampling + masks) ===")
    print(f"Mattes MI before (↑): {mi_before:.6f}")
    print(f"Mattes MI after  (↑): {mi_after:.6f}    Δ = {mi_after - mi_before:+.6f}")
    print("Stop condition:", R.GetOptimizerStopConditionDescription())
    print("Iterations    :", R.GetOptimizerIteration())

    if out_transform_path:
        sitk.WriteTransform(final_tx, out_transform_path)
        print("Saved transform :", out_transform_path)

    if out_warped_path:
        mri_refined = sitk.Resample(mri, ct, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        sitk.WriteImage(mri_refined, out_warped_path)
        print("Saved warped MRI:", out_warped_path)

    return {
        "mi_before": mi_before,
        "mi_after":  mi_after,
        "transform": final_tx,
        "stop":  R.GetOptimizerStopConditionDescription(),
        "iters": R.GetOptimizerIteration(),
    }


# ---------------------------------------------------------------------------
# Example usage (edit paths before running)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CT/MRI rigid registration via SimpleITK")
    parser.add_argument("--mode", choices=["rigid", "refine", "micro"], default="micro",
                        help="Registration mode")
    parser.add_argument("--ct",   required=True, help="Path to CT image (fixed)")
    parser.add_argument("--mri",  required=True, help="Path to MRI image (moving)")
    parser.add_argument("--ct_mask",  default=None, help="Optional CT mask (UInt8)")
    parser.add_argument("--mri_mask", default=None, help="Optional MRI mask (UInt8)")
    parser.add_argument("--out_vol",  default="mri_registered.nii.gz", help="Output warped MRI path")
    parser.add_argument("--out_tfm",  default="transform.tfm", help="Output transform path")
    parser.add_argument("--init_mode", default="GEOMETRY", choices=["GEOMETRY", "MOMENTS"])
    parser.add_argument("--bins",    type=int,   default=32)
    parser.add_argument("--threads", type=int,   default=8)
    args = parser.parse_args()

    if args.mode == "rigid":
        _, info, _ = rigid_ct_fixed_mri_moving(
            ct_path=args.ct, mri_path=args.mri,
            out_moved=args.out_vol, out_tfm=args.out_tfm,
            fixed_mask_path=args.ct_mask, moving_mask_path=args.mri_mask,
            init_mode=args.init_mode, bins=args.bins,
        )
        print(info)

    elif args.mode == "refine":
        results = refine_rigid_mi_jupyter(
            ct_path=args.ct, mri_path=args.mri,
            out_warped_path=args.out_vol, out_transform_path=args.out_tfm,
            fixed_mask_path=args.ct_mask, moving_mask_path=args.mri_mask,
            init_mode=args.init_mode, histogram_bins=args.bins,
        )
        print(results["before"])
        print(results["after"])

    elif args.mode == "micro":
        res = micro_refine_rigid_mi_none_sampling(
            ct_path=args.ct, mri_path=args.mri,
            fixed_mask_path=args.ct_mask, moving_mask_path=args.mri_mask,
            init_mode=args.init_mode, bins=args.bins, threads=args.threads,
            out_warped_path=args.out_vol, out_transform_path=args.out_tfm,
        )
        print(res)
