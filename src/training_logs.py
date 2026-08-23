
import matplotlib.pyplot as plt
import numpy as np
import torch
import skimage
from collections import defaultdict


class MetricMonitor:
    """
    Tracks running averages of named metrics for training progress display.

    Use update() to add values, then str(monitor) to get a formatted summary
    of all metrics.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'value': 0, 'count': 0, 'average': 0})

    def update(self, metric_name, value):
        metric = self.metrics[metric_name]

        metric['value'] += value
        metric['count'] += 1
        metric['average'] = metric['value'] / metric['count']

    def __str__(self):
        return ' | '.join(
            ['{metric_name}: {temp:.{float_precision}f}'.format(metric_name=metric_name, temp=metric['average'], float_precision=self.float_precision) for (metric_name, metric) in self.metrics.items()]
        )


class Visualization:
    """
    Builds and saves training visualizations at specified epochs.

    Holds loss history, per-epoch tensor losses, alpha weights, and images
    (fixed, moving, support, residual, etc.). Exposes methods to generate
    pixelwise loss maps, image reconstructions, loss-over-epoch plots, and
    deformation flow visualizations.
    """

    def __init__(self, losses, tensor_losses, alpha, epoch, images, spatial_size, output_path):
        self.losses = losses
        self.alpha = alpha
        self.epoch = epoch
        self.spatial_size = spatial_size
        self.output_path = output_path

        # All images are in pytorch tensor format [batch, channels, height, width]
        self.moving_image = images.get('moving', None)
        self.fixed_image = images.get('fixed', None)
        self.support_image = images.get('support', None)
        self.residual_image = images.get('residual', None)
        self.reconstructed_image = images.get('reconstructed', None)
        self.moved_affine = images.get('moved_affine', None)
        self.moved_deform = images.get('moved_deform', None)

        # Flow field from deformation network [N, 3] or None
        self.flow = images.get('flow', None)

        # Decomposition losses
        self.loss_mse = tensor_losses.get('mse', None)
        self.loss_grayscale = tensor_losses.get('grayscale', None)
        self.loss_bg_var = tensor_losses.get('bg_var', None)
        self.loss_edge = tensor_losses.get('edge', None)
        # Registration losses
        self.loss_lncc = tensor_losses.get('lncc', None)
        self.loss_lncc_support = tensor_losses.get('lncc_support', None)
        self.loss_flow_tv = tensor_losses.get('flow_tv', None)
        self.loss_jacobian = tensor_losses.get('jacobian', None)


    def reconstruct_pixelwise_losses(self):
        """
        Plot pixelwise losses: mse, grayscale, bg_var, edge, lcc, lcc_support, flow_tv, jacobian.
        """
        h, w = self.spatial_size

        def prepare(arr, *steps):
            for fn in steps:
                arr = fn(arr)
            return arr

        def to_2d(a):
            x = a.squeeze(0)
            return x.mean(axis=0) if x.ndim == 3 else x

        plot_config = [
            ('Reconstruction', self.loss_mse, [lambda a: a.mean(axis=1).squeeze(0), lambda a: a * self.alpha[0]]),
            ('Grayscale', self.loss_grayscale, [lambda a: a.squeeze(0), lambda a: a * self.alpha[1]]),
            ('Background Variance', self.loss_bg_var, [to_2d, lambda a: a * self.alpha[2]]),
            ('Edge', self.loss_edge, [lambda a: a.squeeze(0).squeeze(0), lambda a: a * self.alpha[3]]),
            ('NCC', self.loss_lncc, [lambda a: -a.squeeze(0).squeeze(0), lambda a: a * self.alpha[4]]),
            ('NCC Support', self.loss_lncc_support, [lambda a: -a.squeeze(0).squeeze(0), lambda a: a * self.alpha[5]]),
            ('Flow TV', self.loss_flow_tv, [lambda a: np.zeros((h, w)) if a.ndim == 0 else a.reshape(h, w), lambda a: a * self.alpha[6]]),
            ('Jacobian', self.loss_jacobian, [lambda a: a.reshape(h, w), lambda a: a * self.alpha[7]]),
        ]

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
        for ax, (title, tensor, steps) in zip(axes.flatten(), plot_config):
            arr = prepare(tensor.detach().cpu().numpy(), *steps)
            ax.imshow(arr, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.output_path.replace('.png', f'_{self.epoch}_loss_pixelwise.png'))
        plt.close(fig)


    def reconstruct_images(self):
        """
        Plot image reconstructions: fixed, reconstructed, residual, support, moved_deform, moved_affine, moving.
        """
        def to_hwc(t):
            return t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        plot_config = [
            ('Fixed', self.fixed_image),
            ('Reconstructed', self.reconstructed_image),
            ('Residual', self.residual_image),
            ('Support', self.support_image),
            ('Moved - Deformable', self.moved_deform),
            ('Moved - Affine', self.moved_affine),
            ('Moving', self.moving_image),
        ]

        fig, axes = plt.subplots(nrows=1, ncols=len(plot_config), figsize=(3.5 * len(plot_config), 3))
        for ax, (title, img) in zip(axes, plot_config):
            img = to_hwc(img)
            output_shape = (self.spatial_size[0], self.spatial_size[1], img.shape[2]) if img.ndim == 3 else self.spatial_size
            img = skimage.transform.resize(img, output_shape, anti_aliasing=True)
            img = np.clip((img + 1) * 0.5, 0, 1)
            ax.imshow(img, vmin=0, vmax=1)
            ax.set_xticks(np.arange(0, img.shape[1], 32))
            ax.set_yticks(np.arange(0, img.shape[0], 32))
            ax.grid(True, alpha=0.5, color='black', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.output_path.replace('.png', f'_{self.epoch}_reconstructions.png'))
        plt.close(fig)


    def plot_losses_over_epochs(self):
        """
        Graphs of losses over epochs.
        Row 1: Decompostion losses (mse, grayscale, bg_var, edge)
        Row 2: Registration losses (lncc, lncc_support, flow_tv, jacobian)
        Row 3: Total loss
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4)

        plot_config = [
            ('Reconstruction', [('reconstruction', None)], 0),
            ('Grayscale', [('grayscale', None)], 1),
            ('Background Variance', [('bg_var', None)], 2),
            ('Edge', [('edge', None)], 3),
            ('NCC', [('lncc', 'Local NCC'), ('global_ncc', 'Global NCC'), ('lcc', 'NCC')], 4),
            ('NCC Support', [('lncc_support', 'Local NCC Support'), ('global_ncc_support', 'Global NCC Support'), ('lcc_support', 'NCC Support')], 5),
            ('Flow TV', [('flow_tv', 'Flow TV')], 6),
            ('Jacobian', [('jacobian', None)], 7),
        ]

        axes_row1 = [fig.add_subplot(gs[0, j]) for j in range(4)]
        axes_row2 = [fig.add_subplot(gs[1, j]) for j in range(4)]
        ax_total = fig.add_subplot(gs[2, 1:3])

        for ax, (title, series, i) in zip(axes_row1 + axes_row2, plot_config):
            for key, label in series:
                ax.plot(self.losses[key], label=label)
            ax.set_title(f'{title}, $\\alpha_{i + 1}$={self.alpha[i]:.2f}')
            if any(lbl for _, lbl in series):
                ax.legend(fontsize=8)

        ax_total.plot(self.losses['total'], label='Total')
        ax_total.set_title('Total')
        ax_total.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_path.replace('.png', f'_{self.epoch}_loss_plots.png'))
        plt.close(fig)


    def visualize_flow(self):
        """
        Visualize flow: dx, dy, dz, and 3D magnitude (magnitude is the root sum of squares of dx, dy, dz).
        """
        if self.flow is None:
            return

        h, w = self.spatial_size
        flow_3d = self.flow.detach().cpu().numpy().reshape(h, w, 3)
        dx, dy, dz = flow_3d[:, :, 0], flow_3d[:, :, 1], flow_3d[:, :, 2]
        mag_3d = np.sqrt(dx**2 + dy**2 + dz**2)

        plot_config = [
            ('dx', dx, 'RdBu', lambda a: (-np.abs(a).max(), np.abs(a).max())),  # Red for positive, Blue for negative
            ('dy', dy, 'RdBu', lambda a: (-np.abs(a).max(), np.abs(a).max())),  # Red for positive, Blue for negative
            ('dz', dz, 'RdBu', lambda a: (-np.abs(a).max(), np.abs(a).max())),  # Red for positive, Blue for negative
            ('3D magnitude', mag_3d, 'viridis', lambda a: (None, None)),  # Viridis colormap for magnitude
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (title, arr, cmap, vrange_fn) in zip(axes, plot_config):
            vmin, vmax = vrange_fn(arr)
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f'{title} (min={arr.min():.4f}, max={arr.max():.4f})')  # Show min and max values
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(np.arange(0, w, max(w // 8, 1)))
            ax.set_yticks(np.arange(0, h, max(h // 8, 1)))
            ax.grid(True, alpha=0.5, color='black', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()
        plt.savefig(self.output_path.replace('.png', f'_{self.epoch}_flow.png'))
        plt.close(fig)
