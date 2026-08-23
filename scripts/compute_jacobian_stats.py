#!/usr/bin/env python3
"""
Compute Jacobian determinant statistics for phase 2 deformation results.

This script loads checkpoints from phase 2 (epoch 400), reconstructs the
deformation networks, computes the Jacobian determinant of the combined
affine + deformation field, and outputs statistics (min, max, mean) for
each image pair and for the whole dataset.

Results are saved to a 'rebuttal' directory without overwriting existing results.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from networks import ImpNet
from transforms import get_normalized_coordinates_tensor, apply_affine_transform_pytorch_2d3d
from data import load_and_preprocess_niftii_image, make_image_3channel


def visualize_deformation_grid(flow_2d, det_jac, img_size, output_path, grid_spacing=8):
    """
    Visualize the deformation field using a warped grid and Jacobian map.

    Args:
        flow_2d: 2D flow field [N, 2] (displacement in normalized coords)
        det_jac: Jacobian determinant [N]
        img_size: Image size (H, W assumed equal)
        output_path: Path to save the visualization
        grid_spacing: Spacing between grid lines in pixels
    """
    flow_2d_np = flow_2d.detach().cpu().numpy()
    det_jac_np = det_jac.detach().cpu().numpy()

    # Reshape to 2D
    flow_x = flow_2d_np[:, 0].reshape(img_size, img_size)
    flow_y = flow_2d_np[:, 1].reshape(img_size, img_size)
    det_jac_2d = det_jac_np.reshape(img_size, img_size)

    # Convert flow from normalized [-1,1] to pixel coordinates
    # flow is displacement in normalized coords, need to scale to pixels
    flow_x_pixels = flow_x * (img_size / 2)
    flow_y_pixels = flow_y * (img_size / 2)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Warped grid visualization
    ax = axes[0]
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # Flip y-axis
    ax.set_aspect('equal')
    ax.set_title('Deformation Grid')

    # Draw horizontal lines
    for i in range(0, img_size + 1, grid_spacing):
        x_coords = np.arange(img_size)
        y_coords = np.full(img_size, i)
        if i < img_size:
            # Apply deformation
            x_warped = x_coords + flow_x_pixels[i, :]
            y_warped = y_coords + flow_y_pixels[i, :]
        else:
            x_warped = x_coords
            y_warped = y_coords
        ax.plot(x_warped, y_warped, 'b-', linewidth=0.5, alpha=0.7)

    # Draw vertical lines
    for j in range(0, img_size + 1, grid_spacing):
        y_coords = np.arange(img_size)
        x_coords = np.full(img_size, j)
        if j < img_size:
            # Apply deformation
            x_warped = x_coords + flow_x_pixels[:, j]
            y_warped = y_coords + flow_y_pixels[:, j]
        else:
            x_warped = x_coords
            y_warped = y_coords
        ax.plot(x_warped, y_warped, 'b-', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    # 2. Jacobian determinant map
    ax = axes[1]
    # Create diverging colormap centered at 1
    vmax = max(abs(det_jac_2d.max() - 1), abs(det_jac_2d.min() - 1)) + 1
    vmin = 1 - (vmax - 1)
    im = ax.imshow(det_jac_2d, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f'Jacobian Determinant\n(min={det_jac_2d.min():.3f}, max={det_jac_2d.max():.3f})')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    # 3. Negative Jacobian regions (folding)
    ax = axes[2]
    neg_mask = (det_jac_2d < 0).astype(float)
    num_neg = np.sum(neg_mask)
    neg_frac = num_neg / (img_size * img_size) * 100

    # Show negative regions in red
    ax.imshow(neg_mask, cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Folding Regions (det < 0)\n{int(num_neg)} pixels ({neg_frac:.4f}%)')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_combined_summary(results_df, output_dir, experiment_name):
    """
    Create a combined summary figure showing statistics across all pairs.

    Args:
        results_df: DataFrame with per-pair results
        output_dir: Output directory
        experiment_name: Name of experiment for filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sort by negative jacobian count
    df_sorted = results_df.sort_values('num_negative_jacobians', ascending=False)

    # 1. Bar plot of negative Jacobians per pair
    ax = axes[0, 0]
    labels = [f"{r['subject'].replace('INDD', '')}/{r['slab']}" for _, r in df_sorted.iterrows()]
    values = df_sorted['num_negative_jacobians'].values
    colors = ['red' if v > 0 else 'green' for v in values]
    ax.barh(range(len(labels)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Number of Negative Jacobians')
    ax.set_title('Negative Jacobians per Image Pair')
    ax.invert_yaxis()

    # 2. Mean Jacobian distribution
    ax = axes[0, 1]
    ax.hist(results_df['jacobian_mean'].values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Identity (1.0)')
    ax.axvline(x=results_df['jacobian_mean'].mean(), color='green', linestyle='-', linewidth=2,
               label=f'Mean ({results_df["jacobian_mean"].mean():.4f})')
    ax.set_xlabel('Mean Jacobian Determinant')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mean Jacobians')
    ax.legend()

    # 3. Min/Max range per pair
    ax = axes[1, 0]
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.plot([row['jacobian_min'], row['jacobian_max']], [i, i], 'b-', linewidth=2, alpha=0.6)
        ax.plot(row['jacobian_min'], i, 'ro', markersize=4)
        ax.plot(row['jacobian_max'], i, 'go', markersize=4)
        ax.plot(row['jacobian_mean'], i, 'k^', markersize=4)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Folding threshold')
    ax.axvline(x=1, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Identity')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Jacobian Determinant')
    ax.set_title('Jacobian Range per Pair (red=min, green=max, black=mean)')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # 4. Summary statistics text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    SUMMARY STATISTICS
    ==================

    Total Image Pairs: {len(results_df)}
    Total Points: {results_df['num_points'].sum():,}

    Global Jacobian Statistics:
    ---------------------------
    Mean of Means: {results_df['jacobian_mean'].mean():.6f}
    Std of Means:  {results_df['jacobian_mean'].std():.6f}

    Mean of Mins:  {results_df['jacobian_min'].mean():.6f}
    Mean of Maxs:  {results_df['jacobian_max'].mean():.6f}

    Negative Jacobians:
    -------------------
    Total Count:   {results_df['num_negative_jacobians'].sum():,}
    Total Fraction: {results_df['num_negative_jacobians'].sum() / results_df['num_points'].sum() * 100:.6f}%

    Pairs with 0 negatives: {(results_df['num_negative_jacobians'] == 0).sum()}/{len(results_df)}
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.suptitle(f'Jacobian Statistics Summary: {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'jacobian_summary_figure_{experiment_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def compute_jacobian_determinant(input_coords, output, add_identity=False):
    """
    Compute the Jacobian determinant for each point.

    Args:
        input_coords: Input coordinates tensor [N, 2] with requires_grad=True
        output: Output coordinates tensor [N, 2]
        add_identity: Whether to add identity to the Jacobian

    Returns:
        det_jac: Jacobian determinant for each point [N]
    """
    dim = input_coords.shape[1]
    device = input_coords.device
    n_points = input_coords.shape[0]

    jacobian_matrix = torch.zeros(n_points, dim, dim, device=device)

    for i in range(dim):
        grad_outputs = torch.ones_like(output[:, i])
        grad = torch.autograd.grad(
            output[:, i],
            input_coords,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True
        )[0]
        jacobian_matrix[:, i, :] = grad
        if add_identity:
            jacobian_matrix[:, i, i] += 1.0

    det_jac = torch.det(jacobian_matrix)
    return det_jac


def compute_full_jacobian_determinant(input_coords, flow, affine_2x2):
    """
    Compute the full transformation Jacobian determinant including affine.

    The full transformation is: y = A @ x + b + flow(x)
    The Jacobian is: J = A + ∂flow/∂x

    Args:
        input_coords: Input coordinates tensor [N, 2] with requires_grad=True
        flow: Flow/displacement tensor [N, 2]
        affine_2x2: 2x2 affine matrix (without translation)

    Returns:
        det_jac: Full Jacobian determinant for each point [N]
    """
    dim = input_coords.shape[1]
    device = input_coords.device
    n_points = input_coords.shape[0]

    # Compute Jacobian matrix of flow
    jacobian_matrix = torch.zeros(n_points, dim, dim, device=device)

    for i in range(dim):
        grad_outputs = torch.ones_like(flow[:, i])
        grad = torch.autograd.grad(
            flow[:, i],
            input_coords,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True
        )[0]
        jacobian_matrix[:, i, :] = grad

    # Add the affine matrix to each point's Jacobian
    # J_total = A + J_flow
    affine_2x2_expanded = affine_2x2.unsqueeze(0).expand(n_points, -1, -1)
    jacobian_total = jacobian_matrix + affine_2x2_expanded

    det_jac = torch.det(jacobian_total)
    return det_jac


def load_deformation_network(checkpoint, device):
    """
    Load and reconstruct the deformation network from a checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        device: Device to load the network on

    Returns:
        deformation_network: Loaded ImpNet model
    """
    network_size = checkpoint["network_size"]

    deformation_network = ImpNet(
        input_dim=2,
        output_dim=3,
        hidden_dim=256,
        hidden_n_layers=network_size[0],
        last_layer_weights_small=True,
        is_last_linear=True,
        input_omega_0=5.0,
        hidden_omega_0=5.0,
        input_n_encoding_functions=0
    ).to(device)

    deformation_network.load_state_dict(checkpoint["deformation_network_state_dict"])
    deformation_network.eval()

    return deformation_network


def compute_jacobian_stats_for_checkpoint(checkpoint_path, device, fixed_image=None, moving_image=None):
    """
    Compute Jacobian statistics for a single checkpoint.

    The full transformation Jacobian is computed as:
    J_total = J_affine + J_flow

    where J_affine is the 2x2 affine matrix (constant) and J_flow is the
    Jacobian of the deformation network output w.r.t. input coordinates.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to use for computation
        fixed_image: Optional preloaded fixed image
        moving_image: Optional preloaded moving image

    Returns:
        dict with 'min', 'max', 'mean', 'std', 'neg_frac' (fraction of negative determinants)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    img_size = checkpoint["img_size"]
    flow_scale = checkpoint["flow_scale"]
    A = checkpoint["affine_A"].to(device)
    b = checkpoint["affine_b"].to(device)

    deformation_network = load_deformation_network(checkpoint, device)

    # Create coordinate grids
    coords_init = get_normalized_coordinates_tensor((img_size, img_size), device=device, is_vector=True)

    # We need fixed and moving images to compute the affine sampling grid
    # If not provided, we'll create dummy ones with the right shape
    if fixed_image is None:
        fixed_image = torch.zeros(3, img_size, img_size, device=device)
    if moving_image is None:
        # Need 3D moving image for apply_affine_transform_pytorch_2d3d
        moving_image = torch.zeros(3, 10, img_size, img_size, device=device)

    # Get affine sampling grid
    with torch.no_grad():
        sampling_grid, _ = apply_affine_transform_pytorch_2d3d(
            fixed_image, moving_image, A, b, padding_mode="zeros"
        )
        sampling_grid_flat = sampling_grid.view(-1, 3)

    # Compute flow from deformation network
    coords_deform = coords_init.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        flow_raw, coords_deform = deformation_network(coords_deform, clone=True)
        flow = torch.tanh(flow_raw) * flow_scale

        # Compute Jacobian of deformation-only transformation: y = x + flow(x)
        # J_deform = I + ∂flow/∂x
        # This is what reviewers want - the non-rigid deformation after affine alignment
        flow_2d = flow[:, :2]
        det_jac_deform = compute_jacobian_determinant(coords_deform, flow_2d, add_identity=True)

    det_jac_deform_np = det_jac_deform.detach().cpu().numpy()

    # Count negative Jacobians (folding)
    num_negative = int(np.sum(det_jac_deform_np < 0))
    neg_frac = float(num_negative / len(det_jac_deform_np))

    return {
        # Deformation-only Jacobian stats (what reviewers requested)
        # This is det(I + J_flow), should be ~1 for identity, <0 for folding
        "min": float(np.min(det_jac_deform_np)),
        "max": float(np.max(det_jac_deform_np)),
        "mean": float(np.mean(det_jac_deform_np)),
        "std": float(np.std(det_jac_deform_np)),
        "num_negative": num_negative,
        "negative_fraction": neg_frac,
        "num_points": len(det_jac_deform_np),
        "det_jac_flat": det_jac_deform_np,  # Store full array for aggregate stats
        "img_size": img_size,
        # For visualization
        "flow_2d": flow_2d.detach(),
        "det_jac_tensor": det_jac_deform.detach(),
    }


def find_all_checkpoints(data_dir, epoch=400, experiment_filter=None, subject_filter=None, slab_filter=None):
    """
    Find all checkpoint files for a given epoch.

    Args:
        data_dir: Root data directory
        epoch: Epoch number to look for
        experiment_filter: Optional filter for experiment names (e.g., 'purple_init')
        subject_filter: Optional filter for subject ID (e.g., 'INDD127873')
        slab_filter: Optional filter for slab (e.g., 'slab05S')

    Returns:
        List of (subject, slab, experiment, checkpoint_path) tuples
    """
    checkpoints = []
    data_path = Path(data_dir)

    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("INDD"):
            continue

        subject_id = subject_dir.name

        # Apply subject filter if specified
        if subject_filter and subject_filter not in subject_id:
            continue

        work_dir = subject_dir / "work"

        if not work_dir.exists():
            continue

        for slab_dir in sorted(work_dir.iterdir()):
            if not slab_dir.is_dir() or not slab_dir.name.startswith("slab"):
                continue

            slab_id = slab_dir.name

            # Apply slab filter if specified
            if slab_filter and slab_filter not in slab_id:
                continue

            # Look for INR experiment directories
            for exp_dir in sorted(slab_dir.iterdir()):
                if not exp_dir.is_dir() or not exp_dir.name.startswith("INR"):
                    continue

                exp_name = exp_dir.name

                # Apply filter if specified
                if experiment_filter and experiment_filter not in exp_name:
                    continue

                # Look for checkpoint at specified epoch
                checkpoint_pattern = f"*_{epoch}_checkpoint.pt"
                ckpt_files = list(exp_dir.glob(checkpoint_pattern))

                if ckpt_files:
                    checkpoints.append((subject_id, slab_id, exp_name, ckpt_files[0]))

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Compute Jacobian statistics for deformation results")
    parser.add_argument("--data_dir", type=str,
                        default="/data/cathalye/histo_mri_INR/data_shortlist",
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str,
                        default="/data/cathalye/histo_mri_INR/rebuttal",
                        help="Output directory for results")
    parser.add_argument("--epoch", type=int, default=400,
                        help="Epoch number to analyze")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Filter for experiment name (e.g., 'purple_init', 'manual_init')")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:X or cpu)")
    parser.add_argument("--save_grids", action="store_true",
                        help="Save deformation grid visualizations for each pair")
    parser.add_argument("--grid_spacing", type=int, default=8,
                        help="Spacing between grid lines in pixels")
    parser.add_argument("--subject", type=str, default=None,
                        help="Filter for specific subject ID (e.g., 'INDD127873')")
    parser.add_argument("--slab", type=str, default=None,
                        help="Filter for specific slab (e.g., 'slab05S')")
    args = parser.parse_args()

    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    print(f"Searching for epoch {args.epoch} checkpoints in {args.data_dir}...")
    checkpoints = find_all_checkpoints(args.data_dir, args.epoch, args.experiment, args.subject, args.slab)

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints")

    # Compute Jacobian stats for each checkpoint
    results = []
    all_det_jac = []

    for subject_id, slab_id, exp_name, ckpt_path in checkpoints:
        print(f"Processing {subject_id}/{slab_id}/{exp_name}...")

        try:
            stats = compute_jacobian_stats_for_checkpoint(ckpt_path, device)

            results.append({
                "subject": subject_id,
                "slab": slab_id,
                "experiment": exp_name,
                "checkpoint_path": str(ckpt_path),
                # Deformation-only Jacobian stats (non-rigid stage after affine)
                "jacobian_min": stats["min"],
                "jacobian_max": stats["max"],
                "jacobian_mean": stats["mean"],
                "jacobian_std": stats["std"],
                "num_negative_jacobians": stats["num_negative"],
                "negative_jacobian_fraction": stats["negative_fraction"],
                "num_points": stats["num_points"],
            })

            # Collect all determinants for aggregate stats
            all_det_jac.append(stats["det_jac_flat"])

            print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, "
                  f"Mean: {stats['mean']:.6f}, #Neg: {stats['num_negative']}, "
                  f"Neg%: {stats['negative_fraction']*100:.4f}%")

            # Save deformation grid visualization
            if args.save_grids:
                grid_dir = output_dir / "deformation_grids"
                grid_dir.mkdir(parents=True, exist_ok=True)
                grid_filename = f"{subject_id}_{slab_id}_{exp_name}_deformation_grid.png"
                grid_path = grid_dir / grid_filename
                visualize_deformation_grid(
                    stats["flow_2d"],
                    stats["det_jac_tensor"],
                    stats["img_size"],
                    grid_path,
                    grid_spacing=args.grid_spacing
                )
                print(f"  Grid visualization saved to: {grid_path}")

        except Exception as e:
            print(f"  Error processing checkpoint: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("No results computed!")
        return

    # Create DataFrame and save per-pair results
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_suffix = f"_{args.experiment}" if args.experiment else ""
    csv_filename = f"jacobian_stats_epoch{args.epoch}{exp_suffix}_{timestamp}.csv"
    csv_path = output_dir / csv_filename

    df.to_csv(csv_path, index=False)
    print(f"\nPer-pair results saved to: {csv_path}")

    # Compute and save aggregate statistics
    all_det_jac_concat = np.concatenate(all_det_jac)
    total_negative = int(np.sum(all_det_jac_concat < 0))

    aggregate_stats = {
        "total_pairs": len(results),
        "total_points": len(all_det_jac_concat),
        "global_min": float(np.min(all_det_jac_concat)),
        "global_max": float(np.max(all_det_jac_concat)),
        "global_mean": float(np.mean(all_det_jac_concat)),
        "global_std": float(np.std(all_det_jac_concat)),
        "global_median": float(np.median(all_det_jac_concat)),
        "total_negative_jacobians": total_negative,
        "global_negative_fraction": float(total_negative / len(all_det_jac_concat)),
        "per_pair_mean_of_mins": float(df["jacobian_min"].mean()),
        "per_pair_mean_of_maxs": float(df["jacobian_max"].mean()),
        "per_pair_mean_of_means": float(df["jacobian_mean"].mean()),
        "per_pair_std_of_means": float(df["jacobian_mean"].std()),
        "total_negative_per_pair": int(df["num_negative_jacobians"].sum()),
    }

    # Save aggregate stats
    summary_filename = f"jacobian_summary_epoch{args.epoch}{exp_suffix}_{timestamp}.txt"
    summary_path = output_dir / summary_filename

    with open(summary_path, "w") as f:
        f.write("Jacobian Determinant Statistics Summary\n")
        f.write("(Non-rigid Deformation Stage - After Affine Alignment)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis timestamp: {timestamp}\n")
        f.write(f"Epoch analyzed: {args.epoch}\n")
        f.write(f"Experiment filter: {args.experiment or 'None (all experiments)'}\n")
        f.write(f"Data directory: {args.data_dir}\n\n")

        f.write("Dataset Overview\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total image pairs: {aggregate_stats['total_pairs']}\n")
        f.write(f"Total points analyzed: {aggregate_stats['total_points']:,}\n\n")

        f.write("Global Statistics (All Points Pooled)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Minimum Jacobian:           {aggregate_stats['global_min']:.6f}\n")
        f.write(f"Maximum Jacobian:           {aggregate_stats['global_max']:.6f}\n")
        f.write(f"Mean Jacobian:              {aggregate_stats['global_mean']:.6f}\n")
        f.write(f"Std Dev:                    {aggregate_stats['global_std']:.6f}\n")
        f.write(f"Median:                     {aggregate_stats['global_median']:.6f}\n")
        f.write(f"Number of Negative Jacobians: {aggregate_stats['total_negative_jacobians']:,}\n")
        f.write(f"Negative Jacobian Fraction: {aggregate_stats['global_negative_fraction']*100:.6f}%\n\n")

        f.write("Per-Pair Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean of per-pair minimums:  {aggregate_stats['per_pair_mean_of_mins']:.6f}\n")
        f.write(f"Mean of per-pair maximums:  {aggregate_stats['per_pair_mean_of_maxs']:.6f}\n")
        f.write(f"Mean of per-pair means:     {aggregate_stats['per_pair_mean_of_means']:.6f}\n")
        f.write(f"Std of per-pair means:      {aggregate_stats['per_pair_std_of_means']:.6f}\n\n")

        f.write("Interpretation\n")
        f.write("-" * 40 + "\n")
        f.write("These metrics evaluate the non-rigid deformation stage ONLY\n")
        f.write("(i.e., the deformation network output, independent of affine).\n\n")
        f.write("- Jacobian determinant = 1: Identity (no local deformation)\n")
        f.write("- Jacobian determinant > 1: Local expansion\n")
        f.write("- 0 < Jacobian determinant < 1: Local contraction\n")
        f.write("- Jacobian determinant < 0: Folding (topology violation)\n\n")
        f.write("A good deformation field should have:\n")
        f.write("- Mean close to 1 (minimal volume change on average)\n")
        f.write("- Low standard deviation (smooth deformation)\n")
        f.write("- Zero or very few negative Jacobians (no folding)\n")

    print(f"Summary saved to: {summary_path}")

    # Generate summary figure
    exp_name_str = args.experiment or "all"
    summary_fig_path = visualize_combined_summary(df, output_dir, f"epoch{args.epoch}_{exp_name_str}")
    print(f"Summary figure saved to: {summary_fig_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("JACOBIAN DETERMINANT STATISTICS SUMMARY")
    print("(Non-rigid Deformation Stage - After Affine Alignment)")
    print("=" * 60)
    print(f"\nTotal pairs analyzed: {aggregate_stats['total_pairs']}")
    print(f"Total points: {aggregate_stats['total_points']:,}")
    print(f"\nGlobal statistics (all points pooled):")
    print(f"  Min:    {aggregate_stats['global_min']:.6f}")
    print(f"  Max:    {aggregate_stats['global_max']:.6f}")
    print(f"  Mean:   {aggregate_stats['global_mean']:.6f}")
    print(f"  Std:    {aggregate_stats['global_std']:.6f}")
    print(f"  Median: {aggregate_stats['global_median']:.6f}")
    print(f"  Number of negative Jacobians: {aggregate_stats['total_negative_jacobians']:,}")
    print(f"  Negative fraction: {aggregate_stats['global_negative_fraction']*100:.6f}%")
    print(f"\nPer-pair statistics:")
    print(f"  Mean of minimums: {aggregate_stats['per_pair_mean_of_mins']:.6f}")
    print(f"  Mean of maximums: {aggregate_stats['per_pair_mean_of_maxs']:.6f}")
    print(f"  Mean of means:    {aggregate_stats['per_pair_mean_of_means']:.6f}")


if __name__ == "__main__":
    main()
