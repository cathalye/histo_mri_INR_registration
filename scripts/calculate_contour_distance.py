#!/usr/bin/env python3
"""
Calculate symmetric root mean squared distance (RMSD) between contour images.

This script compares fixed histology contours with moved contours (affine and deform)
to measure registration accuracy.
"""

import argparse
import csv
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import cdist


def load_and_binarize_contour(path, threshold=0.5):
    """
    Load a NIfTI contour image and binarize it using a fixed threshold.

    Args:
        path: Path to NIfTI file
        threshold: Threshold value for binarization (default: 0.5)

    Returns:
        binary_array: Binary numpy array (True for contour points)
        spacing: Image spacing tuple
        origin: Image origin tuple
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Contour file not found: {path}")

    sitk_image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(sitk_image)

    # Handle multi-channel images - take first channel if needed
    if array.ndim > 2 and array.shape[-1] <= 4:
        # Likely a multi-channel image, take first channel
        array = array[..., 0] if array.ndim == 3 else array[:, :, 0]

    # Remove singleton dimensions (e.g., (1, H, W) -> (H, W))
    array = np.squeeze(array)

    # Ensure we have at least 2D
    if array.ndim < 2:
        raise ValueError(f"Array has insufficient dimensions: {array.shape}")

    # Check if array is already binary (only 0s and 1s, or close to it)
    unique_vals = np.unique(array)
    is_binary = len(unique_vals) <= 2 and (
        (np.allclose(unique_vals, [0, 1])) or
        (len(unique_vals) == 1 and (unique_vals[0] == 0 or unique_vals[0] == 1))
    )

    # Normalize to [0, 1] only if not already in that range
    # Don't normalize if already binary or if values are already in [0, 1] range
    array_min = array.min()
    array_max = array.max()

    if is_binary:
        # Already binary, use as-is
        array_normalized = array
    elif array_min >= -0.1 and array_max <= 1.1:
        # Already in [0, 1] range (or close to it), use as-is
        # This handles moved contours that are already normalized
        array_normalized = np.clip(array, 0, 1)
    else:
        # Need to normalize (e.g., if in [-1, 1] or other range)
        if array_max > array_min:
            array_normalized = (array - array_min) / (array_max - array_min)
        else:
            array_normalized = array

    # Binarize using threshold
    binary_array = array_normalized > threshold

    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()

    # Adjust spacing for 2D if needed (remove z-spacing if present)
    if len(spacing) == 3 and array.ndim == 2:
        spacing = spacing[:2]

    return binary_array.astype(np.uint8), spacing, origin


def calculate_symmetric_rmsd(contour1, contour2, spacing):
    """
    Calculate symmetric root mean squared distance (RMSD) between two contours.

    Extracts all contour points and calculates nearest neighbor distances:
    1. Extract all points from contour1 and contour2
    2. For each point in contour1, find nearest point in contour2
    3. For each point in contour2, find nearest point in contour1
    4. Calculate RMSD for both directions and return mean

    This approach correctly handles overlapping contours (distance = 0 for overlapping points).

    Args:
        contour1: Binary numpy array (contour points are True/1)
        contour2: Binary numpy array (contour points are True/1)
        spacing: Image spacing tuple (for physical distance calculation)

    Returns:
        symmetric_rmsd: Mean of RMSD_A_to_B and RMSD_B_to_A
        rmsd_1_to_2: RMSD from contour1 to contour2
        rmsd_2_to_1: RMSD from contour2 to contour1
    """
    # Handle mismatched dimensions - try to match by removing singleton dimensions
    contour1_squeezed = np.squeeze(contour1)
    contour2_squeezed = np.squeeze(contour2)

    if contour1_squeezed.shape != contour2_squeezed.shape:
        raise ValueError(f"Contour dimensions mismatch: {contour1.shape} (squeezed: {contour1_squeezed.shape}) vs {contour2.shape} (squeezed: {contour2_squeezed.shape})")

    # Use squeezed versions
    contour1 = contour1_squeezed
    contour2 = contour2_squeezed

    # Extract contour points (where value > 0)
    points1 = np.argwhere(contour1 > 0)
    points2 = np.argwhere(contour2 > 0)

    # Handle empty contours
    if len(points1) == 0:
        raise ValueError("Contour1 is empty (no contour points found)")
    if len(points2) == 0:
        raise ValueError("Contour2 is empty (no contour points found)")

    # Convert spacing to numpy array for physical distance calculation
    # SimpleITK spacing is (x, y) for 2D or (x, y, z) for 3D
    # NumPy arrays use (row, col) convention, so we need to reverse
    # For 2D: spacing (x, y) -> numpy (row, col) = (y, x)
    # For 3D: spacing (x, y, z) -> numpy (depth, row, col) = (z, y, x)
    if len(spacing) == 2:
        spacing_array = np.array([spacing[1], spacing[0]])  # (y, x) for (row, col)
    elif len(spacing) == 3:
        spacing_array = np.array([spacing[2], spacing[1], spacing[0]])  # (z, y, x) for (depth, row, col)
    else:
        # Fallback: use ones if spacing doesn't match
        spacing_array = np.ones(len(contour1.shape))

    # Convert pixel coordinates to physical coordinates using spacing
    # Multiply each coordinate by its corresponding spacing
    points1_physical = points1.astype(np.float64) * spacing_array
    points2_physical = points2.astype(np.float64) * spacing_array

    # Calculate pairwise distances between all points
    # cdist returns a matrix of shape (len(points1), len(points2))
    distances_matrix = cdist(points1_physical, points2_physical, metric='euclidean')

    # For each point in contour1, find the distance to nearest point in contour2
    distances_1_to_2 = np.min(distances_matrix, axis=1)  # Shape: (len(points1),)

    # For each point in contour2, find the distance to nearest point in contour1
    distances_2_to_1 = np.min(distances_matrix, axis=0)  # Shape: (len(points2),)

    # Calculate RMSD for both directions
    rmsd_1_to_2 = np.sqrt(np.mean(distances_1_to_2 ** 2))
    rmsd_2_to_1 = np.sqrt(np.mean(distances_2_to_1 ** 2))

    # Symmetric RMSD is the mean of both directions
    symmetric_rmsd = (rmsd_1_to_2 + rmsd_2_to_1) / 2.0

    return symmetric_rmsd, rmsd_1_to_2, rmsd_2_to_1


def main():
    parser = argparse.ArgumentParser(
        description="Calculate symmetric RMSD between fixed and moved contours"
    )
    parser.add_argument('--fixed_contour', type=str, required=True,
                        help='Path to fixed histology contour NIfTI file')
    parser.add_argument('--moved_affine', type=str, required=True,
                        help='Path to moved affine contour NIfTI file')
    parser.add_argument('--moved_deform', type=str, required=True,
                        help='Path to moved deform contour NIfTI file')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to output CSV file')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Binarization threshold (default: 0.2, lower for interpolated contours)')
    parser.add_argument('--subject_id', type=str, required=True,
                        help='Subject ID (e.g., INDD100013)')
    parser.add_argument('--slab_id', type=str, required=True,
                        help='Slab ID (e.g., slab09S)')

    args = parser.parse_args()

    print(f"Calculating contour distances for {args.subject_id} / {args.slab_id}")
    print(f"  Fixed contour: {args.fixed_contour}")
    print(f"  Moved affine: {args.moved_affine}")
    print(f"  Moved deform: {args.moved_deform}")
    print(f"  Threshold: {args.threshold}")
    print()

    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    try:
        # Load and binarize contours
        print("Loading fixed contour...")
        fixed_contour, fixed_spacing, _ = load_and_binarize_contour(args.fixed_contour, args.threshold)
        print(f"  Shape: {fixed_contour.shape}, Spacing: {fixed_spacing}")
        print(f"  Contour points: {np.sum(fixed_contour > 0)}")

        print("Loading moved affine contour...")
        moved_affine, affine_spacing, _ = load_and_binarize_contour(args.moved_affine, args.threshold)
        print(f"  Shape: {moved_affine.shape}, Spacing: {affine_spacing}")
        print(f"  Contour points: {np.sum(moved_affine > 0)}")

        print("Loading moved deform contour...")
        moved_deform, deform_spacing, _ = load_and_binarize_contour(args.moved_deform, args.threshold)
        print(f"  Shape: {moved_deform.shape}, Spacing: {deform_spacing}")
        print(f"  Contour points: {np.sum(moved_deform > 0)}")
        print()

        # Calculate symmetric RMSD for affine
        print("Calculating symmetric RMSD (affine)...")
        affine_rmsd, affine_1_to_2, affine_2_to_1 = calculate_symmetric_rmsd(
            fixed_contour, moved_affine, fixed_spacing
        )
        print(f"  RMSD (fixed -> affine): {affine_1_to_2:.4f}")
        print(f"  RMSD (affine -> fixed): {affine_2_to_1:.4f}")
        print(f"  Symmetric RMSD: {affine_rmsd:.4f}")
        print()

        # Calculate symmetric RMSD for deform
        print("Calculating symmetric RMSD (deform)...")
        deform_rmsd, deform_1_to_2, deform_2_to_1 = calculate_symmetric_rmsd(
            fixed_contour, moved_deform, fixed_spacing
        )
        print(f"  RMSD (fixed -> deform): {deform_1_to_2:.4f}")
        print(f"  RMSD (deform -> fixed): {deform_2_to_1:.4f}")
        print(f"  Symmetric RMSD: {deform_rmsd:.4f}")
        print()

        # Write results to CSV
        csv_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if file is new
            if not csv_exists:
                writer.writerow(['subject_id', 'slab_id', 'affine_rmsd', 'deform_rmsd'])

            # Write data row
            writer.writerow([
                args.subject_id,
                args.slab_id,
                f'{affine_rmsd:.6f}',
                f'{deform_rmsd:.6f}'
            ])

        print(f"Results saved to: {args.output_csv}")
        print("Done!")

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
