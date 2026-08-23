#!/usr/bin/env python3
"""
Generate violin plots of Jacobian determinant distributions for each image pair.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from networks import ImpNet
from transforms import get_normalized_coordinates_tensor, apply_affine_transform_pytorch_2d3d


def compute_jacobian_determinant(input_coords, output, add_identity=False):
    """Compute the Jacobian determinant for each point."""
    dim = input_coords.shape[1]
    device = input_coords.device
    n_points = input_coords.shape[0]
    
    jacobian_matrix = torch.zeros(n_points, dim, dim, device=device)
    
    for i in range(dim):
        grad_outputs = torch.ones_like(output[:, i])
        grad = torch.autograd.grad(
            output[:, i], input_coords, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=True
        )[0]
        jacobian_matrix[:, i, :] = grad
        if add_identity:
            jacobian_matrix[:, i, i] += 1.0
    
    return torch.det(jacobian_matrix)


def load_deformation_network(checkpoint, device):
    """Load and reconstruct the deformation network from a checkpoint."""
    network_size = checkpoint["network_size"]
    
    deformation_network = ImpNet(
        input_dim=2, output_dim=3, hidden_dim=256,
        hidden_n_layers=network_size[0], last_layer_weights_small=True,
        is_last_linear=True, input_omega_0=5.0, hidden_omega_0=5.0,
        input_n_encoding_functions=0
    ).to(device)
    
    deformation_network.load_state_dict(checkpoint["deformation_network_state_dict"])
    deformation_network.eval()
    return deformation_network


def get_jacobian_distribution(checkpoint_path, device):
    """Get the full Jacobian distribution for a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    img_size = checkpoint["img_size"]
    flow_scale = checkpoint["flow_scale"]
    
    deformation_network = load_deformation_network(checkpoint, device)
    coords_init = get_normalized_coordinates_tensor((img_size, img_size), device=device, is_vector=True)
    coords_deform = coords_init.clone().detach().requires_grad_(True)
    
    with torch.enable_grad():
        flow_raw, coords_deform = deformation_network(coords_deform, clone=True)
        flow = torch.tanh(flow_raw) * flow_scale
        flow_2d = flow[:, :2]
        det_jac = compute_jacobian_determinant(coords_deform, flow_2d, add_identity=True)
    
    return det_jac.detach().cpu().numpy()


def find_checkpoints(data_dir, epoch=400, experiment_filter=None):
    """Find all checkpoint files."""
    checkpoints = []
    data_path = Path(data_dir)
    
    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("INDD"):
            continue
        
        work_dir = subject_dir / "work"
        if not work_dir.exists():
            continue
        
        for slab_dir in sorted(work_dir.iterdir()):
            if not slab_dir.is_dir() or not slab_dir.name.startswith("slab"):
                continue
            
            for exp_dir in sorted(slab_dir.iterdir()):
                if not exp_dir.is_dir() or not exp_dir.name.startswith("INR"):
                    continue
                
                if experiment_filter and experiment_filter not in exp_dir.name:
                    continue
                
                ckpt_files = list(exp_dir.glob(f"*_{epoch}_checkpoint.pt"))
                if ckpt_files:
                    label = f"{subject_dir.name.replace('INDD', '')}/{slab_dir.name.replace('slab', '')}"
                    checkpoints.append((label, ckpt_files[0]))
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Generate Jacobian violin plots")
    parser.add_argument("--data_dir", type=str, 
                        default="/data/cathalye/histo_mri_INR/data_shortlist")
    parser.add_argument("--output_dir", type=str,
                        default="/data/cathalye/histo_mri_INR/rebuttal")
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--experiment", type=str, default="purple_init")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.data_dir, args.epoch, args.experiment)
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Collect Jacobian distributions
    all_distributions = []
    labels = []
    
    for label, ckpt_path in checkpoints:
        print(f"Processing {label}...")
        det_jac = get_jacobian_distribution(ckpt_path, device)
        all_distributions.append(det_jac)
        labels.append(label)
    
    # Sort by median Jacobian for better visualization
    medians = [np.median(d) for d in all_distributions]
    sorted_indices = np.argsort(medians)[::-1]
    all_distributions = [all_distributions[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Subsample for faster plotting (violin plots with 65k points per violin are slow)
    subsample_size = 5000
    subsampled = [np.random.choice(d, size=min(subsample_size, len(d)), replace=False) 
                  for d in all_distributions]
    
    parts = ax.violinplot(subsampled, positions=range(len(labels)), 
                          showmeans=True, showmedians=True, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        # Color based on negative fraction
        neg_frac = np.sum(all_distributions[i] < 0) / len(all_distributions[i])
        if neg_frac == 0:
            pc.set_facecolor('green')
            pc.set_alpha(0.7)
        elif neg_frac < 0.01:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
        else:
            pc.set_facecolor('orange')
            pc.set_alpha(0.7)
    
    # Customize mean and median lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    
    # Add horizontal lines
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Identity (det=1)')
    ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Folding threshold (det=0)')
    
    # Add statistics annotations
    for i, (label, dist) in enumerate(zip(labels, all_distributions)):
        neg_count = np.sum(dist < 0)
        neg_pct = neg_count / len(dist) * 100
        # Add text above each violin
        ax.text(i, ax.get_ylim()[1] * 0.95, f'{neg_pct:.2f}%', 
                ha='center', va='top', fontsize=7, rotation=90)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Jacobian Determinant', fontsize=12)
    ax.set_xlabel('Subject/Slab', fontsize=12)
    ax.set_title(f'Jacobian Determinant Distribution per Image Pair\n(Epoch {args.epoch}, {args.experiment})\n'
                 f'Green=0% neg, Blue=<1% neg, Orange=≥1% neg | Red line=mean, Black line=median',
                 fontsize=11)
    
    # Set y-axis limits to focus on the main distribution
    y_min = min(np.percentile(d, 0.5) for d in all_distributions)
    y_max = max(np.percentile(d, 99.5) for d in all_distributions)
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(min(y_min - margin, -0.5), y_max + margin)
    
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"jacobian_violin_epoch{args.epoch}_{args.experiment}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nViolin plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nPer-pair summary (sorted by median):")
    print("-" * 60)
    for label, dist in zip(labels, all_distributions):
        neg_frac = np.sum(dist < 0) / len(dist) * 100
        print(f"{label:20s} | median={np.median(dist):.4f} | mean={np.mean(dist):.4f} | neg={neg_frac:.4f}%")


if __name__ == "__main__":
    main()
