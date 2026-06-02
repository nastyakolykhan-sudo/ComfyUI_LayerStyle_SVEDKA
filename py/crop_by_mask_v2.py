import torch
import numpy as np
from PIL import Image

from .imagefunc import log, tensor2pil, pil2tensor, image2mask, min_bounding_rect, max_inscribed_rect, mask_area
from .imagefunc import num_round_up_to_multiple, draw_rect


def _bbox_from_mask_np(mask_np, detect):
    """
    mask_np: uint8 [H, W] 0-255
    Returns (x, y, w, h) or (0, 0, 0, 0) if empty.
    """
    if detect == "mask_area":
        locs = np.where(mask_np > 127)
        if len(locs[0]) == 0:
            return 0, 0, 0, 0
        y1, x1 = int(locs[0].min()), int(locs[1].min())
        y2, x2 = int(locs[0].max()), int(locs[1].max())
        return x1, y1, x2 - x1, y2 - y1
    else:
        m_pil = Image.fromarray(mask_np, mode='L')
        if detect == "min_bounding_rect":
            return min_bounding_rect(m_pil)
        else:
            return max_inscribed_rect(m_pil)


def _letterbox(img_pil, output_w, output_h, pad_color=(0, 0, 0)):
    """
    Fit img_pil into output_w x output_h maintaining aspect ratio.
    Centers content, pads remainder with pad_color.
    Returns (result_pil, scale, offset_x, offset_y, content_w, content_h)
    """
    src_w, src_h = img_pil.size
    scale = min(output_w / src_w, output_h / src_h)
    content_w = max(1, int(src_w * scale))
    content_h = max(1, int(src_h * scale))
    resized = img_pil.resize((content_w, content_h), Image.LANCZOS)
    result = Image.new(img_pil.mode, (output_w, output_h), color=pad_color)
    offset_x = (output_w - content_w) // 2
    offset_y = (output_h - content_h) // 2
    result.paste(resized, (offset_x, offset_y))
    return result, scale, offset_x, offset_y, content_w, content_h


class CropByMaskV2:

    def __init__(self):
        self.NODE_NAME = 'CropByMask V2'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_data", "box_preview")
    FUNCTION = 'crop_by_mask_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_by_mask_v2(self, image, mask, invert_mask, detect,
                        top_reserve, bottom_reserve,
                        left_reserve, right_reserve,
                        output_width, output_height,
                        round_to_multiple,
                        crop_box=None):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if invert_mask:
            mask = 1 - mask

        canvas_height, canvas_width = image.shape[1], image.shape[2]
        n_images = image.shape[0]
        n_masks  = mask.shape[0]
        max_batch = max(n_images, n_masks)

        ret_images = []
        ret_masks  = []
        crop_data  = []  # list of per-frame dicts for RestoreCropBox

        # mask as uint8 numpy for fast bbox detection
        mask_np = (mask.numpy() * 255).clip(0, 255).astype(np.uint8)  # [N, H, W]

        for i in range(max_batch):
            img_i  = image[min(i, n_images - 1)]   # [H, W, 3]
            mask_i = mask_np[min(i, n_masks - 1)]   # [H, W] uint8

            if crop_box is not None:
                # Use provided fixed box for all frames
                x1, y1, x2, y2 = crop_box
            else:
                x, y, w, h = _bbox_from_mask_np(mask_i, detect)

                if w == 0 or h == 0:
                    # Empty mask — use full canvas
                    x, y, w, h = 0, 0, canvas_width, canvas_height

                x1 = max(x - left_reserve, 0)
                y1 = max(y - top_reserve, 0)
                x2 = min(x + w + right_reserve, canvas_width)
                y2 = min(y + h + bottom_reserve, canvas_height)

                if round_to_multiple != 'None':
                    multiple = int(round_to_multiple)
                    rw = num_round_up_to_multiple(x2 - x1, multiple)
                    rh = num_round_up_to_multiple(y2 - y1, multiple)
                    x1 = x1 - (rw - (x2 - x1)) // 2
                    y1 = y1 - (rh - (y2 - y1)) // 2
                    x2 = x1 + rw
                    y2 = y1 + rh

            # Clamp to canvas
            x1c = max(x1, 0)
            y1c = max(y1, 0)
            x2c = min(x2, canvas_width)
            y2c = min(y2, canvas_height)

            # Crop image and mask as PIL for letterbox
            img_pil  = Image.fromarray(
                (img_i.numpy() * 255).clip(0, 255).astype(np.uint8), mode='RGB')
            mask_pil = Image.fromarray(mask_i, mode='L')

            cropped_img  = img_pil.crop((x1c, y1c, x2c, y2c))
            cropped_mask = mask_pil.crop((x1c, y1c, x2c, y2c))

            # Letterbox into output_width × output_height
            lb_img,  scale, off_x, off_y, cw, ch = _letterbox(
                cropped_img,  output_width, output_height, pad_color=(0, 0, 0))
            lb_mask, _,     _,     _,     _,  _  = _letterbox(
                cropped_mask, output_width, output_height, pad_color=0)

            ret_images.append(pil2tensor(lb_img))
            ret_masks.append(image2mask(lb_mask))

            crop_data.append({
                "box":       (x1c, y1c, x2c, y2c),   # original crop coords on canvas
                "scale":     scale,                    # scale applied during letterbox
                "offset_x":  off_x,                   # content offset inside output frame
                "offset_y":  off_y,
                "content_w": cw,                      # content size inside output frame
                "content_h": ch,
            })

        # Preview — draw per-frame box on first mask
        preview_pil = Image.fromarray(mask_np[0], mode='L').convert('RGB')
        first_box = crop_data[0]["box"]
        preview_pil = draw_rect(preview_pil, first_box[0], first_box[1],
                                first_box[2] - first_box[0], first_box[3] - first_box[1],
                                line_color="#00F000",
                                line_width=(first_box[2] - first_box[0] +
                                            first_box[3] - first_box[1]) // 200)

        log(f"{self.NODE_NAME} Processed {max_batch} image(s).", message_type='finish')
        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks,  dim=0),
            crop_data,
            pil2tensor(preview_pil),
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V2": CropByMaskV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V2": "LayerUtility: CropByMask V2"
}
