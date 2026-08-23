from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RegistrationLosses:
    """
    Loss modules for registration training.
    """
    mse: torch.nn.Module
    grayscale: torch.nn.Module
    bg_var: torch.nn.Module
    edge: torch.nn.Module
    ncc: torch.nn.Module
    lncc: torch.nn.Module
    reg: torch.nn.Module
    flow_tv: torch.nn.Module

    @classmethod
    def prepare(cls, device, n_channels):
        """
        Create and return RegistrationLosses instances for the given device and channels.
        """
        return cls(
            mse=MSELoss(is_tensor=True).to(device),
            grayscale=GrayscaleLoss(is_tensor=True).to(device),
            bg_var=BackgroundVarianceLoss(is_tensor=True).to(device),
            edge=EdgeMatchingLoss(is_tensor=True).to(device),
            ncc=NCCLoss(is_tensor=False).to(device),
            lncc=LNCCLoss(win=(32, 32), n_channels=n_channels, is_tensor=True).to(device),
            reg=JacobianLossCoords(add_identity=False, is_tensor=True).to(device),
            flow_tv=FlowTVLoss(is_tensor=True).to(device),
        )


def get_alpha(start_alpha, end_alpha, epoch, total_epochs, phase1_epochs=None, interpolate_full_indices=None):
    """
    Compute alpha weights by linearly interpolating from start to end.

    If phase1_epochs is provided:
    - Alphas in interpolate_full_indices: interpolate from start to end over the full training (both phases)
    - Other alphas: stay at start during phase 1, interpolate from start to end during phase 2 only

    Args:
        start_alpha: Starting weights for each loss at epoch 0
        end_alpha: Ending weights for each loss at final epoch
        epoch: Current epoch
        total_epochs: Total number of epochs
        phase1_epochs: Number of epochs in phase 1 (optional). If None, uses total_epochs for all.
        interpolate_full_indices: Indices of alphas that interpolate over full training (both phases).
            Default: range(6) (alpha1-6). Alpha7 (flow_tv) and alpha8 (jacobian) only interpolate over phase 2.

    Returns:
        List of alpha weights for current epoch
    """

    # TODO: edit this based on final alpha schedule after experiments

    phase1_epochs=None
    if interpolate_full_indices is None:
        interpolate_full_indices = set(range(6))  # alpha1-6 full, alpha7 (jac) alpha8 (flow_tv) phase 2 only
    else:
        interpolate_full_indices = set(interpolate_full_indices)

    if phase1_epochs is None:
        # No phase separation, interpolate over all epochs
        progress_full = epoch / total_epochs
        progress_phase2 = epoch / total_epochs
    else:
        progress_full = epoch / total_epochs
        if epoch < phase1_epochs:
            progress_phase2 = 0.0
        else:
            phase2_epochs = total_epochs - phase1_epochs
            progress_phase2 = (epoch - phase1_epochs) / phase2_epochs

    new_alpha = []
    for i in range(len(start_alpha)):
        if i in interpolate_full_indices:
            progress = progress_full
        else:
            progress = progress_phase2
        new_alpha.append(start_alpha[i] + progress * (end_alpha[i] - start_alpha[i]))

    return new_alpha


class MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss.

    Encourages the support + residual image to match the fixed image.
    """
    def __init__(self, is_tensor=False):
        super(MSELoss, self).__init__()
        self.is_tensor = is_tensor

    def forward(self, y_true, y_pred, mask=None):
        loss = (y_true - y_pred)**2

        if mask is not None:
            # Handle different mask shapes
            if len(mask.shape) == 2:  # (H, W)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif len(mask.shape) == 3:  # (C, H, W)
                mask = mask.unsqueeze(0)  # (1, C, H, W)

            # Expand mask to match loss shape if needed
            if mask.shape != loss.shape:
                mask = mask.expand_as(loss)

            loss = loss * mask.float()

        if self.is_tensor:
            return loss
        else:
            if mask is not None:
                # Average only over masked pixels
                return loss.sum() / mask.float().sum().clamp(min=1)
            else:
                return torch.mean(loss)


class GrayscaleLoss(torch.nn.Module):
    """
    Encourage support to be grayscale (R==G==B) like MRI.

    Penalizes differences between color channels, pushing the support image
    toward a grayscale representation where R, G, and B values are equal
    at each pixel.
    """
    def __init__(self, is_tensor=False):
        super(GrayscaleLoss, self).__init__()
        self.is_tensor = is_tensor

    def forward(self, image, mask=None):
        # image shape: [1, 3, H, W] or [N, H*W, 3] depending on representation
        # mask shape: [1, 1, H, W] or None

        # Handle both image tensor formats
        if image.dim() == 4:
            # Standard image format: [N, C, H, W]
            r, g, b = image[:, 0], image[:, 1], image[:, 2]
        elif image.dim() == 3:
            # Coordinate-based format: [N, H*W, C]
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Compute pairwise squared differences between channels
        loss = (r - g)**2 + (g - b)**2 + (r - b)**2

        # Apply mask if provided (only compute loss within tissue regions)
        if mask is not None:
            # Ensure mask has compatible shape
            if mask.dim() == 4 and mask.shape[1] == 1:
                # [N, 1, H, W] -> [N, H, W]
                mask = mask.squeeze(1)
            elif mask.dim() == 2:
                # [H, W] -> [1, H, W]
                mask = mask.unsqueeze(0)

            loss = loss * mask

        if self.is_tensor:
            return loss
        else:
            if mask is not None:
                # Average only over masked pixels
                return loss.sum() / mask.sum().clamp(min=1)
            else:
                return loss.mean()


class BackgroundVarianceLoss(torch.nn.Module):
    """
    Penalize variance of background pixels in the support image.

    Ensures that the area outside the mask (background) in the support
    image is constant by minimizing the variance of those pixels. A zero variance
    means all background pixels have the same value.

    Together, the edge matching and the background variance losses prevent the
    support network from encoding extraneous tissue visible in the full MRI coronal
    slice but absent from the histology section.

    The loss computes Var(support[~mask]) for each channel and averages them.
    """
    def __init__(self, is_tensor=False):
        super(BackgroundVarianceLoss, self).__init__()
        self.is_tensor = is_tensor

    def forward(self, support_image, mask):
        # Handle different mask shapes
        if mask.dim() == 2:  # (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:  # (N, H, W)
            mask = mask.unsqueeze(1)  # (N, 1, H, W)

        # Invert mask to get background (0 -> 1, 1 -> 0)
        bg_mask = 1.0 - mask.float()

        # Expand mask to match support image channels
        bg_mask = bg_mask.expand_as(support_image)  # [N, C, H, W]

        # Count background pixels
        num_bg_pixels = bg_mask.sum()

        if num_bg_pixels < 2:
            # Not enough background pixels to compute variance
            return torch.tensor(0.0, device=support_image.device, requires_grad=True)

        # Extract background pixels
        bg_pixels = support_image * bg_mask

        # Compute mean of background pixels
        bg_mean = bg_pixels.sum() / num_bg_pixels

        # Compute variance: E[(x - mean)^2] for background pixels
        variance = ((bg_pixels - bg_mean * bg_mask) ** 2 * bg_mask).sum() / num_bg_pixels

        if self.is_tensor:
            # Return per-pixel squared deviation from mean (for visualization)
            return (support_image - bg_mean) ** 2 * bg_mask
        else:
            return variance


class EdgeMatchingLoss(torch.nn.Module):
    """
    High-Frequency Edge Matching - anchors sharp structural edges to the Support image.

    Forces the Support image to match the high-frequency edges of the original
    (grayscale) image using Sobel gradients. This ensures fine details like
    Nissl bodies and myelin fibers are captured in the Support, not the Residual.

    Together, the edge matching and the background variance losses prevent the
    support network from encoding extraneous tissue visible in the full MRI coronal
    slice but absent from the histology section.

    L_edge = || grad(Support) - grad(Original_gray) ||_1
    """
    def __init__(self, is_tensor=False):
        super(EdgeMatchingLoss, self).__init__()
        self.is_tensor = is_tensor

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Register as buffers (not parameters, but will move to device with module)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradients(self, image):
        # Compute Sobel gradients for a grayscale image
        # image shape: [N, 1, H, W]
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        return grad_x, grad_y

    def _to_grayscale(self, image):
        # Convert RGB to grayscale
        # image shape: [N, 3, H, W]
        # Standard luminosity method
        if image.shape[1] == 3:
            return 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        else:
            return image

    def forward(self, support_image, original_image, mask=None):
        # support_image shape: [N, C, H, W]
        # original_image shape: [N, C, H, W]

        # Convert both to grayscale
        support_gray = self._to_grayscale(support_image)
        original_gray = self._to_grayscale(original_image)

        # Compute Sobel gradients
        support_grad_x, support_grad_y = self._compute_gradients(support_gray)
        original_grad_x, original_grad_y = self._compute_gradients(original_gray)

        # L1 loss between gradients
        loss_x = torch.abs(support_grad_x - original_grad_x)
        loss_y = torch.abs(support_grad_y - original_grad_y)
        loss = loss_x + loss_y  # [N, 1, H, W]

        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:  # (H, W)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif mask.dim() == 3:  # (N, H, W)
                mask = mask.unsqueeze(1)  # (N, 1, H, W)
            elif mask.dim() == 4 and mask.shape[1] != 1:
                mask = mask[:, 0:1]  # Take first channel

            loss = loss * mask.float()

        if self.is_tensor:
            return loss
        else:
            if mask is not None:
                return loss.sum() / mask.float().sum().clamp(min=1)
            else:
                return loss.mean()


class LNCCLoss(torch.nn.Module):
    """
    Local normalized cross-correlation loss. Computed locally over a window of size win.
    Default window size is 32x32.
    """
    def __init__(self,
                 win=(32, 32),
                 n_channels=3,
                 is_tensor=False):
        super(LNCCLoss, self).__init__()

        self.is_tensor = is_tensor
        self.win = win
        self.n_channels = n_channels
        self.win_size = np.prod(win)*self.n_channels
        self.ndims = len(win)

        self.conv = getattr(torch.nn, 'Conv%dd' % self.ndims)(in_channels=n_channels, out_channels=1,
                                                              kernel_size=self.win, stride=1, padding='same',
                                                              padding_mode='replicate', bias=False)

        # kernel is not trainable, so no gradient needed
        with torch.no_grad():
            # initialize the convolution kernel with all weights set to 1
            torch.nn.init.ones_(self.conv.weight)

        for param in self.conv.parameters():
            # kernel weights are not trainable and won't be updated
            param.requires_grad = False

    def forward(self, y_true, y_pred, mask=None):
        # dividing by win_size calculates the mean image intensity over the window size
        true_sum = self.conv(y_true) / self.win_size
        pred_sum = self.conv(y_pred) / self.win_size

        # subtract the mean intensity from the image (centers the image around 0)
        true_cent = y_true - true_sum
        pred_cent = y_pred - pred_sum

        numerator = self.conv(true_cent * pred_cent)
        # XXX: should this be squared? - not taking the square root of the denominator so keep the numerator squared
        # If I take the square root of the denominator and remove this square, then the losses go to nan
        numerator = numerator * numerator

        # self.conv(true_cent * true_cent) is the local variance of the true image (sum of squares of differences from the mean)
        # self.conv(pred_cent * pred_cent) is the local variance of the predicted image
        # denominator = torch.sqrt(self.conv(true_cent * true_cent) * self.conv(pred_cent * pred_cent))
        denominator = self.conv(true_cent * true_cent) * self.conv(pred_cent * pred_cent)

        cc = (numerator + 1e-6) / (denominator + 1e-6)
        # XXX: do we want to clamp or keep the negative values?
        cc = torch.clamp(cc, 0, 1)
        # This does not work
        # cc = (cc + 1) * 0.5

        if mask is not None:
            if len(mask.shape) == 2: # (H, W)
                mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            elif len(mask.shape) == 3: # (C, H, W)
                mask = mask.unsqueeze(0) # (1, C, H, W)

            cc = cc * mask

            mask = mask.expand_as(cc)
            cc = cc * mask.float()

        if self.is_tensor==True:
            return -cc
        else:
            # average only over the masked pixels
            if mask is not None:
                cc = cc.sum() / mask.float().sum().clamp(min=1)
                return -cc
            else:
                return -torch.mean(cc)


class NCCLoss(torch.nn.Module):
    """
    Global normalized cross-correlation loss. Gives us a single value for the
    similarity between the two images.
    """
    def __init__(self, is_tensor=False):
        super(NCCLoss, self).__init__()
        self.is_tensor = is_tensor

    def forward(self, y_true, y_pred):
        # taking the .mean() at the end, gives us the global score for the image
        numerator = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
        denominator = y_true.std() * y_pred.std()

        cc = (numerator + 1e-6) / (denominator + 1e-6) # range is [-1.0, 1.0]
        # XXX: what about cases where the images are rotated wrt each other?
        cc = torch.clamp(cc, 0, 1)
        # This does not work
        # cc = (cc + 1) * 0.5

        # high values are good but we want to minimize the loss, so we negate the value
        # cc is a scalar so torch.mean() is not actually needed here
        if self.is_tensor==True:
            return -cc
        else:
            return -torch.mean(cc)


class JacobianLossCoords(torch.nn.Module):
    """
    Jacobian loss for the coordinates.

    Penalizes the local deformation of the transformation.
    """
    def __init__(self,
                 add_identity=False,
                 is_tensor=False):

        super(JacobianLossCoords, self).__init__()

        # add_identity is used to denote whether we are using the displacement field
        # or the transformed coordinates
        #
        # if add_identity is True, we expect the the input coords and the displacement field
        # as inputs to the forward function. The transformed coordinates are obtained by adding
        # the displacement field to the original coordinates (identity transform).
        #
        # if add_identity is False, we expect the the input coords and the transformed coordinates
        # as inputs to the forward function.

        self.add_identity = add_identity
        self.is_tensor = is_tensor

    def forward(self, input_coords, output, mask=None):
        jac = self.compute_jacobian_matrix(input_coords, output, add_identity=self.add_identity)
        # Determinant of the Jacobian matrix is a measure of the local deformation
        # of the transformation.
        # det > 1 means the transformation is expanding -> loss is negative
        # det < 1 means the transformation is contracting -> loss is positive but less than 1
        # det < 0 means the transformation is folding - WE DON'T WANT THIS -> loss is more than 1
        loss = 1 - torch.det(jac)

        # taking the absolute value ensures larger deformations are
        # penalized irrespective of the sign
        loss = torch.abs(loss)

        if mask is not None:
            if len(mask.shape) > 1: # (H, W) or (C, H, W)
                mask_flat = mask.flatten().to(loss.device)
            else:
                mask_flat = mask

            loss = loss * mask_flat

        if self.is_tensor==True:
            return loss
        else:
            if mask is not None:
                loss = loss.sum() / mask_flat.float().sum().clamp(min=1)
                return loss
            else:
                return torch.mean(loss)

    def compute_jacobian_matrix(self, input_coords, output, add_identity=False):
        dim = input_coords.shape[1]
        jacobian_matrix = torch.zeros(input_coords.shape[0], dim, dim)

        for i in range(dim):
            jacobian_matrix[:, i, :] = self.gradient(input_coords, output[:, i])
            if add_identity:
                jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])

        return jacobian_matrix

    def gradient(self, input_coords, output, grad_outputs=None):
        grad_outputs = torch.ones_like(output)
        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return grad


class FlowTVLoss(torch.nn.Module):
    """
    Total Variation loss for deformation flow fields.

    Encourages spatial smoothness by penalizing differences between
    neighboring flow vectors. This helps ensure that deformations are
    locally consistent (similar magnitude and direction for nearby pixels).
    """
    def __init__(self, is_tensor=False):
        super(FlowTVLoss, self).__init__()
        self.is_tensor = is_tensor

    def forward(self, flow, spatial_size):
        h, w = spatial_size

        # Reshape flow from [N, 3] to [1, 3, H, W] (treat 3D flow as 3-channel image)
        flow_reshaped = flow.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Calculate spatial gradients
        # Horizontal differences (along width)
        diff_w = flow_reshaped[:, :, :, 1:] - flow_reshaped[:, :, :, :-1]  # [1, 3, H, W-1]
        # Vertical differences (along height)
        diff_h = flow_reshaped[:, :, 1:, :] - flow_reshaped[:, :, :-1, :]  # [1, 3, H-1, W]

        # Compute L2 norm of differences across the 3 flow dimensions
        # This penalizes the magnitude of change in the flow vector
        tv_w = torch.sqrt(torch.sum(diff_w ** 2, dim=1, keepdim=True) + 1e-8)  # [1, 1, H, W-1]
        tv_h = torch.sqrt(torch.sum(diff_h ** 2, dim=1, keepdim=True) + 1e-8)  # [1, 1, H-1, W]

        # Pad to match dimensions for unified [H, W] magnitude
        tv_w_s = tv_w.squeeze()  # [H, W-1]
        tv_h_s = tv_h.squeeze()  # [H-1, W]
        tv_w_padded = F.pad(tv_w_s, (0, 1), mode='constant', value=0)   # [H, W]
        tv_h_padded = F.pad(tv_h_s, (0, 0, 0, 1), mode='constant', value=0)  # [H, W]
        magnitude = torch.sqrt(tv_w_padded ** 2 + tv_h_padded ** 2 + 1e-8)
        loss = magnitude.mean()

        if self.is_tensor:
            return magnitude
        else:
            return loss
