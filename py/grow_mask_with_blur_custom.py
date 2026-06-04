import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from itertools import groupby
from tqdm import tqdm

from .imagefunc import log


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(radius: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian kernel; sigma = radius / 3."""
    sigma = max(radius / 3.0, 0.01)
    size = int(radius) * 2 + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_batch_gpu(batch: torch.Tensor, blur_list: list) -> torch.Tensor:
    """
    Per-frame separable Gaussian blur on GPU.
    batch    : [N, H, W] float32
    blur_list: radius per frame (last value repeats if shorter than N)
    Returns  : [N, H, W] on same device.
    """
    device = batch.device
    n = batch.shape[0]
    indexed = [(i, blur_list[i] if i < len(blur_list) else blur_list[-1]) for i in range(n)]
    out_frames = []

    for r, group in groupby(indexed, key=lambda x: round(x[1], 2)):
        indices = [g[0] for g in group]
        if r <= 0:
            out_frames.extend([(i, batch[i]) for i in indices])
            continue
        k1d = _gaussian_kernel_1d(r, device)
        kh  = k1d.view(1, 1, 1, -1)
        kv  = k1d.view(1, 1, -1, 1)
        pad = len(k1d) // 2
        sub = batch[indices].unsqueeze(1)           # [B,1,H,W]
        sub = F.pad(sub, (pad, pad, 0, 0), mode='reflect')
        sub = F.conv2d(sub, kh)
        sub = F.pad(sub, (0, 0, pad, pad), mode='reflect')
        sub = F.conv2d(sub, kv)
        sub = sub.squeeze(1)
        out_frames.extend([(indices[j], sub[j]) for j in range(len(indices))])

    out_frames.sort(key=lambda x: x[0])
    return torch.stack([f for _, f in out_frames], dim=0)


def _make_diamond_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """Diamond-shaped morphology kernel of given radius (|x|+|y| <= radius)."""
    size = 2 * radius + 1
    k = torch.zeros(size, size, dtype=torch.float32, device=device)
    for y in range(size):
        for x in range(size):
            if abs(y - radius) + abs(x - radius) <= radius:
                k[y, x] = 1.0
    return k


def _dilate_erode_gpu(output: torch.Tensor, expand: int,
                      tapered_corners: bool) -> torch.Tensor:
    """
    Single-pass morphological dilation or erosion using a kernel sized to `expand`.
    output: [1, 1, H, W]  (already on GPU)
    Returns [1, 1, H, W].
    """
    r = abs(expand)
    device = output.device

    if tapered_corners:
        kernel = _make_diamond_kernel(r, device)   # diamond footprint
    else:
        kernel = torch.ones(2 * r + 1, 2 * r + 1, dtype=torch.float32, device=device)

    pad = r
    if expand > 0:
        # Dilation: max-pool with the kernel shape
        # Use kornia if available for exact kernel, else max_pool2d (square only)
        try:
            import kornia.morphology as morph
            return morph.dilation(output, kernel)
        except Exception:
            return F.max_pool2d(output, kernel_size=2*r+1, stride=1, padding=pad)
    else:
        # Erosion: invert → dilate → invert
        try:
            import kornia.morphology as morph
            return morph.erosion(output, kernel)
        except Exception:
            inv = 1.0 - output
            inv = F.max_pool2d(inv, kernel_size=2*r+1, stride=1, padding=pad)
            return 1.0 - inv


MAX_RESOLUTION = 8192
main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

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
                "flip_input":      ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                }),
                # Temporal smoothing over the blur_radius list.
                # 0 = off; higher = smoother transitions between frames.
                "blur_smoothing": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
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
    RETURN_TYPES  = ("MASK", "MASK",)
    RETURN_NAMES  = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"

    def expand_mask(self, mask, expand, incremental_expandrate, tapered_corners,
                    flip_input, blur_radius, blur_smoothing, lerp_alpha, decay_factor,
                    fill_holes=False):

        alpha = lerp_alpha
        decay = decay_factor

        if flip_input:
            mask = 1.0 - mask

        # --- Normalise all per-frame parameters to lists ---
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        n_frames = growmask.shape[0]

        if isinstance(blur_radius, (list, tuple)):
            blur_list = [float(v) for v in blur_radius]
        else:
            blur_list = [float(blur_radius)] * n_frames

        if isinstance(expand, (list, tuple)):
            expand_list = [int(v) for v in expand]
        else:
            expand_list = [int(expand)] * n_frames

        # --- Temporal smoothing on blur values ---
        # Smooths out frame-to-frame jumps so the blur doesn't flicker.
        if blur_smoothing > 0 and len(blur_list) > 1:
            arr = np.array(blur_list, dtype=np.float32)
            arr = scipy.ndimage.gaussian_filter1d(arr, sigma=blur_smoothing)
            blur_list = arr.tolist()

        # --- Per-frame expand + lerp/decay loop ---
        out = []
        previous_output = None

        for i, m in enumerate(tqdm(growmask, desc="Expanding/Contracting Mask")):
            current_expand = expand_list[i] if i < len(expand_list) else expand_list[-1]
            output = m.unsqueeze(0).unsqueeze(0).to(main_device)

            if current_expand != 0 and output.max() > 0:
                output = _dilate_erode_gpu(output, current_expand, tapered_corners)

            output = output.squeeze(0).squeeze(0)

            if fill_holes:
                # GPU flood-fill approximation: label connected components,
                # fill any component not touching the border.
                # Falls back to scipy (CPU) which is reliable.
                binary_np = (output > 0).cpu().numpy()
                filled_np = scipy.ndimage.binary_fill_holes(binary_np)
                output = torch.from_numpy(filled_np.astype(np.float32)).to(output.device)

            # lerp/decay: useful for temporal smoothing when expand is a scalar.
            # When driving from a per-frame list these should stay at defaults (1.0/1.0).
            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                if output.max() > 0:
                    output = output / output.max()

            previous_output = output
            out.append(output.cpu())

        # --- Batched GPU blur ---
        stacked = torch.stack(out, dim=0)   # [N, H, W]
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
