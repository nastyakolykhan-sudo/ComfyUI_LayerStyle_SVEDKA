import torch
import copy
import numpy as np
import scipy.ndimage
from PIL import Image, ImageFilter
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, step_color, step_value, chop_image_v2, BLEND_MODES


def _expand_mask_fast(mask: torch.Tensor, grow: int, blur: int) -> torch.Tensor:
    """Optimized expand_mask: single dilation/erosion call with a sized diamond footprint
    instead of iterating a 3x3 kernel once per pixel of grow radius."""
    growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    if grow != 0:
        radius = abs(grow)
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        footprint = (np.abs(x) + np.abs(y) <= radius).astype(np.uint8)
        for m in growmask:
            output = m.numpy()
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=footprint)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=footprint)
            out.append(torch.from_numpy(output))
    else:
        out = [m for m in growmask]
    for idx, tensor in enumerate(out):
        if blur > 0:
            pil_image = tensor2pil(tensor.cpu().detach())
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
            out[idx] = pil2tensor(pil_image)
        else:
            out[idx] = tensor.unsqueeze(0)
    return torch.cat(out, dim=0)


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
                "background_image": ("IMAGE", ),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (chop_mode_list,),
                "light_color": ("STRING", {"default": "#FFBF30"}),
                "glow_color": ("STRING", {"default": "#FE0000"}),
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
            opacity=100, brightness=5, glow_range=48, blur=25,
            layer_mask=None
            ):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []

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
        blur_list = blur if isinstance(blur, (list, tuple)) else [blur] * max_batch
        opacity_list = opacity if isinstance(opacity, (list, tuple)) else [opacity] * max_batch

        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')
            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')
            _glow_range = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
            _brightness = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
            _blur = blur_list[i] if i < len(blur_list) else blur_list[-1]
            _opacity = opacity_list[i] if i < len(opacity_list) else opacity_list[-1]
            blur_factor = _blur / 20.0
            grow = _glow_range
            for x in range(_brightness):
                blur_val = int(grow * blur_factor)
                _color = step_color(glow_color, light_color, _brightness, x)
                glow_mask = _expand_mask_fast(image2mask(_mask), grow, blur_val)
                color_image = Image.new("RGB", _layer.size, color=_color)
                alpha = tensor2pil(glow_mask).convert('L')
                _glow = chop_image_v2(_canvas, color_image, blend_mode, int(step_value(1, _opacity, _brightness, x)))
                _canvas.paste(_glow.convert('RGB'), mask=alpha)
                grow = grow - int(_glow_range / _brightness)
            _canvas.paste(_layer, mask=_mask)
            ret_images.append(pil2tensor(_canvas))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast": OuterGlowV2Fast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast": "LayerStyle: OuterGlow V2 Fast SVEDKA"
}
