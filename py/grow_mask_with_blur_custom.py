import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from tqdm import tqdm
from PIL import ImageFilter

from .imagefunc import log, tensor2pil, pil2tensor


def _gaussian_kernel_1d(radius: float, device: torch.device) -> torch.Tensor:
    """Build a 1-D Gaussian kernel for the given radius (sigma = radius / 3)."""
    sigma = max(radius / 3.0, 0.01)
    size = int(radius) * 2 + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_batch_gpu(batch: torch.Tensor, blur_list: list) -> torch.Tensor:
    """
    Apply per-frame Gaussian blur on GPU using separable convolution.
    batch: [N, H, W] float32 on any device
    blur_list: list of float radii, one per frame (or shorter — last value repeats)
    Returns [N, H, W] on same device.
    """
    device = batch.device
    n = batch.shape[0]
    out_frames = []

    # Group frames by radius to minimise kernel rebuilds
    from itertools import groupby
    indexed = [(i, blur_list[i] if i < len(blur_list) else blur_list[-1]) for i in range(n)]

    for r, group in groupby(indexed, key=lambda x: round(x[1], 2)):
        indices = [g[0] for g in group]
        if r <= 0:
            out_frames.extend([(i, batch[i]) for i in indices])
            continue

        k1d = _gaussian_kernel_1d(r, device)
        # Horizontal kernel: [1, 1, 1, kW]
        kh = k1d.view(1, 1, 1, -1)
        # Vertical kernel:   [1, 1, kH, 1]
        kv = k1d.view(1, 1, -1, 1)
        pad = len(k1d) // 2

        sub = batch[indices].unsqueeze(1)           # [B, 1, H, W]
        sub = F.pad(sub, (pad, pad, 0, 0), mode='reflect')
        sub = F.conv2d(sub, kh)
        sub = F.pad(sub, (0, 0, pad, pad), mode='reflect')
        sub = F.conv2d(sub, kv)
        sub = sub.squeeze(1)                        # [B, H, W]

        out_frames.extend([(indices[j], sub[j]) for j in range(len(indices))])

    # Re-sort to original frame order
    out_frames.sort(key=lambda x: x[0])
    return torch.stack([f for _, f in out_frames], dim=0)

try:
    import kornia.morphology as morph
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

MAX_RESOLUTION = 8192

# Use GPU if available, else CPU
main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrowMaskWithBlurCustom:

    def __init__(self):
        self.NODE_NAME = 'GrowMaskWithBlur CUSTOM'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {
                    "default": 0,
                    "min": -MAX_RESOLUTION,
                    "max": MAX_RESOLUTION,
                    "step": 1,
                }),
                "incremental_expandrate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                }),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                }),
                "lerp_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "decay_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = '😺dzNodes/LayerUtility'
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"

    def expand_mask(self, mask, expand, incremental_expandrate, tapered_corners,
                    flip_input, blur_radius, lerp_alpha, decay_factor,
                    fill_holes=False):

        alpha = lerp_alpha
        decay = decay_factor

        if flip_input:
            mask = 1.0 - mask

        # --- Normalise blur_radius to a per-frame list ---
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        n_frames = growmask.shape[0]

        if isinstance(blur_radius, (list, tuple)):
            blur_list = [float(v) for v in blur_radius]
        else:
            blur_list = [float(blur_radius)] * n_frames

        out = []
        previous_output = None
        current_expand = expand

        for i, m in enumerate(tqdm(growmask, desc="Expanding/Contracting Mask")):
            output = m.unsqueeze(0).unsqueeze(0).to(main_device)

            if abs(round(current_expand)) > 0 and output.max() > 0:
                if HAS_KORNIA:
                    if tapered_corners:
                        kernel = torch.tensor(
                            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                            dtype=torch.float32, device=output.device)
                    else:
                        kernel = torch.ones(3, 3, dtype=torch.float32, device=output.device)

                    for _ in range(abs(round(current_expand))):
                        if current_expand < 0:
                            output = morph.erosion(output, kernel)
                        else:
                            output = morph.dilation(output, kernel)
                else:
                    # Fallback: scipy binary dilation/erosion on CPU
                    np_mask = output.squeeze().cpu().numpy()
                    struct = np.array([[0,1,0],[1,1,1],[0,1,0]]) if tapered_corners \
                             else np.ones((3,3))
                    for _ in range(abs(round(current_expand))):
                        if current_expand < 0:
                            np_mask = scipy.ndimage.binary_erosion(np_mask, struct).astype(np.float32)
                        else:
                            np_mask = scipy.ndimage.binary_dilation(np_mask, struct).astype(np.float32)
                    output = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0).to(main_device)

            output = output.squeeze(0).squeeze(0)

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)

            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = scipy.ndimage.binary_fill_holes(output_np)
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)

            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                if output.max() > 0:
                    output = output / output.max()

            previous_output = output
            out.append(output.cpu())

        # --- Apply per-frame blur (GPU batched) ---
        stacked = torch.stack(out, dim=0)   # [N, H, W] on CPU
        any_blur = any(r > 0 for r in blur_list)
        if any_blur:
            gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            result = _gaussian_blur_batch_gpu(stacked.to(gpu), blur_list).cpu()
        else:
            result = stacked

        log(f"{self.NODE_NAME} Processed {n_frames} frame(s).", message_type='finish')
        return (result, 1.0 - result,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: GrowMaskWithBlur CUSTOM": GrowMaskWithBlurCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GrowMaskWithBlur CUSTOM": "LayerUtility: GrowMaskWithBlur CUSTOM",
}
