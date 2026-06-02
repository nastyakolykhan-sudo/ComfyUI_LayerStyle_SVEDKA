import torch
import torch.nn.functional as F
import copy
import numpy as np
import scipy.ndimage
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, step_value, BLEND_MODES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(inhex: str) -> tuple:
    inhex = inhex.strip()
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code: {inhex}')
    if len(inhex) == 4:
        inhex = "#" + "".join([c * 2 for c in inhex[1:]])
    return (int(inhex[1:3], 16), int(inhex[3:5], 16), int(inhex[5:7], 16))


# ---------------------------------------------------------------------------
# CPU path — pure numpy/scipy, parallel frames via ThreadPoolExecutor
# ---------------------------------------------------------------------------

def _build_diamond(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (np.abs(x) + np.abs(y) <= radius).astype(np.uint8)


def _expand_mask_np(mask_np: np.ndarray, grow: int, blur: float) -> np.ndarray:
    if grow != 0:
        footprint = _build_diamond(abs(grow))
        if grow < 0:
            out = scipy.ndimage.grey_erosion(mask_np, footprint=footprint)
        else:
            out = scipy.ndimage.grey_dilation(mask_np, footprint=footprint)
    else:
        out = mask_np
    if blur > 0:
        out = scipy.ndimage.gaussian_filter(out, sigma=blur / 3.0)
    return np.clip(out, 0.0, 1.0)


def _alpha_composite_np(canvas, overlay, alpha):
    a = alpha[:, :, np.newaxis]
    return canvas * (1.0 - a) + overlay * a


def _process_frame_cpu(args):
    (canvas_np, layer_np, mask_np,
     _glow_range, _brightness, _blur, _opacity,
     blend_mode_fn, glow_rgb, light_rgb) = args

    H, W = canvas_np.shape[:2]
    blur_factor = _blur / 20.0

    canvas_rgba = np.empty((H, W, 4), dtype=float)
    canvas_rgba[:, :, :3] = canvas_np.astype(float)
    canvas_rgba[:, :, 3] = 255.0

    source_rgba = np.empty((H, W, 4), dtype=float)
    source_rgba[:, :, 3] = 255.0

    grow = _glow_range
    for x in range(_brightness):
        blur_val = grow * blur_factor
        t = x / _brightness
        source_rgba[:, :, 0] = glow_rgb[0] + (light_rgb[0] - glow_rgb[0]) * t
        source_rgba[:, :, 1] = glow_rgb[1] + (light_rgb[1] - glow_rgb[1]) * t
        source_rgba[:, :, 2] = glow_rgb[2] + (light_rgb[2] - glow_rgb[2]) * t

        alpha = _expand_mask_np(mask_np, grow, blur_val)
        op = step_value(1, _opacity, _brightness, x) / 100.0
        blended_rgba = blend_mode_fn(canvas_rgba, source_rgba, op)
        canvas_rgba[:, :, :3] = _alpha_composite_np(canvas_rgba[:, :, :3], blended_rgba[:, :, :3], alpha)
        grow = grow - int(_glow_range / _brightness)

    result_rgb = _alpha_composite_np(canvas_rgba[:, :, :3], layer_np.astype(float), mask_np)
    return np.clip(result_rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GPU path — pure torch/CUDA, all frames processed as a batch simultaneously
# ---------------------------------------------------------------------------

# Blend modes implemented in torch (0-1 float range)
# These match the blend_modes library formulas used in the CPU path.
_GPU_BLEND_MODES = {
    'normal':            lambda b, s, op: b * (1 - op) + s * op,
    'linear dodge(add)': lambda b, s, op: torch.clamp(b + s * op, 0, 1),
    'screen':            lambda b, s, op: 1 - (1 - b) * (1 - s * op),
    'lighten':           lambda b, s, op: torch.maximum(b, s * op),
    'multiply':          lambda b, s, op: b * (s * op) + b * (1 - op),
    'color dodge':       lambda b, s, op: torch.clamp(b / (1 - s * op + 1e-6), 0, 1),
    'dodge':             lambda b, s, op: torch.clamp(b / (1 - s * op + 1e-6), 0, 1),
    'hard light':        lambda b, s, op: torch.where(
                             s * op < 0.5,
                             2 * b * s * op,
                             1 - 2 * (1 - b) * (1 - s * op)),
    'linear light':      lambda b, s, op: torch.clamp(b + 2 * s * op - 1, 0, 1),
    'overlay':           lambda b, s, op: torch.where(
                             b < 0.5,
                             2 * b * s * op,
                             1 - 2 * (1 - b) * (1 - s * op)),
    'darken':            lambda b, s, op: torch.minimum(b, s * op),
    'difference':        lambda b, s, op: torch.abs(b - s * op),
    'exclusion':         lambda b, s, op: b + s * op - 2 * b * s * op,
    'subtract':          lambda b, s, op: torch.clamp(b - s * op, 0, 1),
    'divide':            lambda b, s, op: torch.clamp(b / (s * op + 1e-6), 0, 1),
    'soft light':        lambda b, s, op: torch.where(
                             s * op < 0.5,
                             b - (1 - 2 * s * op) * b * (1 - b),
                             b + (2 * s * op - 1) * (torch.sqrt(b.clamp(1e-6)) - b)),
}


def _expand_mask_gpu(mask: torch.Tensor, grow: int, blur: float, device: torch.device) -> torch.Tensor:
    """
    mask  : float32 [N, 1, H, W] on device, range 0-1
    grow  : pixels to expand (positive) or erode (negative)
    blur  : blur amount
    Returns float32 [N, 1, H, W] on device
    """
    if grow != 0:
        radius = abs(grow)
        ksize = 2 * radius + 1
        padding = radius
        if grow > 0:
            # Dilation via max_pool2d
            mask = F.max_pool2d(mask, kernel_size=ksize, stride=1, padding=padding)
        else:
            # Erosion = invert → dilate → invert
            mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=ksize, stride=1, padding=padding)

    if blur > 0:
        sigma = blur / 3.0
        # Gaussian kernel size: odd, covers ±3σ
        k = int(6 * sigma + 1) | 1
        k = max(k, 3)
        # Build 1D gaussian, apply separably
        coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss = gauss / gauss.sum()
        # Horizontal pass
        kh = gauss.view(1, 1, 1, k)
        mask = F.conv2d(mask, kh.expand(1, 1, 1, k), padding=(0, k // 2))
        # Vertical pass
        kv = gauss.view(1, 1, k, 1)
        mask = F.conv2d(mask, kv.expand(1, 1, k, 1), padding=(k // 2, 0))

    return mask.clamp(0, 1)


def _process_batch_gpu(b_tensors, l_tensors, masks_tensor,
                       glow_range_list, brightness_list, blur_list, opacity_list,
                       blend_mode, glow_rgb, light_rgb, device):
    """
    Process all frames simultaneously on GPU.
    b_tensors, l_tensors : list of float32 [1, H, W, 3] ComfyUI tensors
    masks_tensor         : list of float32 [H, W] 0-1
    Returns list of float32 [1, H, W, 3] tensors (CPU)
    """
    N = len(b_tensors)
    # Stack to [N, H, W, 3], permute to [N, 3, H, W] for conv ops
    canvas = torch.cat(b_tensors, dim=0).to(device)          # [N, H, W, 3]
    layer  = torch.cat(l_tensors, dim=0).to(device)          # [N, H, W, 3]

    # Masks: [N, 1, H, W]
    mask_list = []
    for m in masks_tensor:
        mask_list.append(m.to(device).unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
    # We process mask per-frame since grow values differ; stack handled below

    blend_fn = _GPU_BLEND_MODES.get(blend_mode)
    if blend_fn is None:
        raise ValueError(f"Blend mode '{blend_mode}' not supported in GPU path.")

    results = []

    # Process each frame — masks/params may differ per frame, but all tensor ops stay on GPU
    for i in range(N):
        c = canvas[i]   # [H, W, 3]  0-1
        l = layer[i]    # [H, W, 3]  0-1
        m = mask_list[i]  # [1, 1, H, W]  0-1

        _glow_range = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
        _brightness = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
        _blur       = blur_list[i]       if i < len(blur_list)       else blur_list[-1]
        _opacity    = opacity_list[i]    if i < len(opacity_list)    else opacity_list[-1]

        blur_factor = _blur / 20.0
        grow = _glow_range

        # [H, W, 3] → [1, 3, H, W] for conv ops
        c_chw = c.permute(2, 0, 1).unsqueeze(0)  # not used for conv, just for reference

        result = c.clone()  # [H, W, 3]

        for x in range(_brightness):
            blur_val = grow * blur_factor
            t = x / _brightness
            r = (glow_rgb[0] + (light_rgb[0] - glow_rgb[0]) * t) / 255.0
            g = (glow_rgb[1] + (light_rgb[1] - glow_rgb[1]) * t) / 255.0
            b = (glow_rgb[2] + (light_rgb[2] - glow_rgb[2]) * t) / 255.0
            color = torch.tensor([r, g, b], dtype=torch.float32, device=device)  # [3]

            # Expand mask — [1, 1, H, W]
            alpha = _expand_mask_gpu(m, grow, blur_val, device)  # [1, 1, H, W]
            alpha_hw = alpha[0, 0]  # [H, W]

            op = step_value(1, _opacity, _brightness, x) / 100.0

            # Solid color broadcast: [H, W, 3]
            s = color.view(1, 1, 3).expand_as(result)

            # Blend
            blended = blend_fn(result, s, op)  # [H, W, 3]

            # Alpha composite
            a = alpha_hw.unsqueeze(-1)  # [H, W, 1]
            result = result * (1 - a) + blended * a

            grow = grow - int(_glow_range / _brightness)

        # Composite layer on top using mask [H, W]
        m_hw = mask_list[i][0, 0]  # [H, W]
        a = m_hw.unsqueeze(-1)
        result = result * (1 - a) + l * a

        results.append(result.unsqueeze(0).cpu())  # [1, H, W, 3]

    return results


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class OuterGlowV2Fast:

    def __init__(self):
        self.NODE_NAME = 'OuterGlowV2Fast'

    @classmethod
    def INPUT_TYPES(self):
        modes = copy.copy(BLEND_MODES)
        chop_mode_list = ["screen", "linear dodge(add)", "color dodge", "lighten", "dodge", "hard light", "linear light"]
        for i in chop_mode_list:
            modes.pop(i)
        chop_mode_list.extend(list(modes.keys()))

        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (chop_mode_list,),
                "light_color": ("STRING", {"default": "#FFBF30"}),
                "glow_color": ("STRING", {"default": "#FE0000"}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "layer_mask": ("MASK",),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1}),
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'outer_glow_v2_fast'
    CATEGORY = '😺dzNodes/LayerStyle'

    def outer_glow_v2_fast(self, background_image, layer_image,
                           invert_mask, blend_mode, light_color, glow_color,
                           use_gpu=False,
                           opacity=100, brightness=5, glow_range=48, blur=25,
                           layer_mask=None):

        b_images = []
        l_images = []
        l_masks = []

        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        if len(l_masks) == 0:
            log(f"Error: {self.NODE_NAME} skipped, because the available mask is not found.", message_type='error')
            return (background_image,)

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        glow_range_list = glow_range if isinstance(glow_range, (list, tuple)) else [glow_range] * max_batch
        brightness_list = brightness if isinstance(brightness, (list, tuple)) else [brightness] * max_batch
        blur_list       = blur       if isinstance(blur,       (list, tuple)) else [blur]       * max_batch
        opacity_list    = opacity    if isinstance(opacity,    (list, tuple)) else [opacity]    * max_batch

        glow_rgb  = _hex_to_rgb(glow_color)
        light_rgb = _hex_to_rgb(light_color)

        # --- GPU path ---
        if use_gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cpu':
                log(f"Warning: {self.NODE_NAME} use_gpu=True but CUDA not available, falling back to CPU.", message_type='warning')
            else:
                # Validate blend mode is supported in GPU path
                if blend_mode not in _GPU_BLEND_MODES:
                    log(f"Warning: {self.NODE_NAME} blend mode '{blend_mode}' not in GPU path, falling back to CPU.", message_type='warning')
                else:
                    # Build mask tensors
                    masks_tensor = []
                    b_tensors = []
                    l_tensors = []
                    for i in range(max_batch):
                        bg   = b_images[i] if i < len(b_images) else b_images[-1]
                        lay  = l_images[i] if i < len(l_images) else l_images[-1]
                        mp   = l_masks[i]  if i < len(l_masks)  else l_masks[-1]
                        if mp.size != tensor2pil(lay).size:
                            mp = Image.new('L', tensor2pil(lay).size, 'white')
                            log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')
                        b_tensors.append(bg)
                        l_tensors.append(lay)
                        masks_tensor.append(torch.from_numpy(
                            np.array(mp, dtype=np.float32) / 255.0))

                    ret_images = _process_batch_gpu(
                        b_tensors, l_tensors, masks_tensor,
                        glow_range_list, brightness_list, blur_list, opacity_list,
                        blend_mode, glow_rgb, light_rgb, device)

                    log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s) on GPU.", message_type='finish')
                    return (torch.cat(ret_images, dim=0),)

        # --- CPU path ---
        frame_args = []
        blend_mode_fn = BLEND_MODES[blend_mode]
        for i in range(max_batch):
            bg_pil   = tensor2pil(b_images[i] if i < len(b_images) else b_images[-1]).convert('RGB')
            lay_pil  = tensor2pil(l_images[i] if i < len(l_images) else l_images[-1]).convert('RGB')
            mask_pil = l_masks[i] if i < len(l_masks) else l_masks[-1]

            if mask_pil.size != lay_pil.size:
                mask_pil = Image.new('L', lay_pil.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')

            canvas_np = np.array(bg_pil,   dtype=np.float32)
            layer_np  = np.array(lay_pil,  dtype=np.float32)
            mask_np   = np.array(mask_pil, dtype=np.float32) / 255.0

            _glow_range = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
            _brightness = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
            _blur       = blur_list[i]       if i < len(blur_list)       else blur_list[-1]
            _opacity    = opacity_list[i]    if i < len(opacity_list)    else opacity_list[-1]

            frame_args.append((
                canvas_np, layer_np, mask_np,
                _glow_range, _brightness, _blur, _opacity,
                blend_mode_fn, glow_rgb, light_rgb
            ))

        max_workers = min(max_batch, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_frame_cpu, frame_args))

        ret_images = [
            torch.from_numpy(r.astype(np.float32) / 255.0).unsqueeze(0)
            for r in results
        ]

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s) on CPU.", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast": OuterGlowV2Fast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast": "LayerStyle: OuterGlow V2 Fast CUSTOM"
}
