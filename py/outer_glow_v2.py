import torch
import copy
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, step_color, step_value, expand_mask, chop_image_v2, chop_mode_v2, BLEND_MODES


class OuterGlowV2:

    def __init__(self):
        self.NODE_NAME = 'OuterGlowV2'

    @classmethod
    def INPUT_TYPES(self):

        modes = copy.copy(BLEND_MODES)
        chop_mode_list = ["screen", "linear dodge(add)", "color dodge", "lighten", "dodge", "hard light", "linear light"]
        for i in chop_mode_list:
            modes.pop(i)
        chop_mode_list.extend(list(modes.keys()))

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_list,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),  # 迭代
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),  # 扩张
                "light_color": ("STRING", {"default": "#FFBF30"}),  # 光源中心颜色
                "glow_color": ("STRING", {"default": "#FE0000"}),  # 辉光外围颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'outer_glow_v2'
    CATEGORY = '😺dzNodes/LayerStyle'

    def outer_glow_v2(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  brightness, glow_range, blur, light_color, glow_color,
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
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            # preprocess
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')
            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')
            glow_range_list = glow_range if isinstance(glow_range, (list, tuple)) else [glow_range] * max_batch
            brightness_list = brightness if isinstance(brightness, (list, tuple)) else [brightness] * max_batch
            blur_list = blur if isinstance(blur, (list, tuple)) else [blur] * max_batch
            opacity_list = opacity if isinstance(opacity, (list, tuple)) else [opacity] * max_batch
            
            glow_range = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
            brightness = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
            blur = blur_list[i] if i < len(blur_list) else blur_list[-1]
            opacity = opacity_list[i] if i < len(opacity_list) else opacity_list[-1]
            blur_factor = blur / 20.0
            grow = glow_range
            for x in range(brightness):
                blur_val = int(grow * blur_factor)
                _color = step_color(glow_color, light_color, brightness, x)
                glow_mask = expand_mask(image2mask(_mask), grow, blur_val)
                color_image = Image.new("RGB", _layer.size, color=_color)
                alpha = tensor2pil(glow_mask).convert('L')
                _glow = chop_image_v2(_canvas, color_image, blend_mode, int(step_value(1, opacity, brightness, x)))
                _canvas.paste(_glow.convert('RGB'), mask=alpha)
                grow = grow - int(glow_range / brightness)
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerStyle: OuterGlow V2": OuterGlowV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: OuterGlow V2": "LayerStyle: OuterGlow V2 SVEDKA"
}