import torch

from .imagefunc import log, tensor2pil, pil2tensor, mask2image, image2mask, gaussian_blur, min_bounding_rect, max_inscribed_rect, mask_area
from .imagefunc import num_round_up_to_multiple, draw_rect



class CropByMaskV2:

    def __init__(self):
        self.NODE_NAME = 'CropByMask V2'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_by_mask_v2(self, image, mask, invert_mask, detect,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve, round_to_multiple,
                     crop_box=None
                     ):

        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if invert_mask:
            mask = 1 - mask
        for m in mask:
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        canvas_width, canvas_height = tensor2pil(l_images[0]).convert('RGB').size

        if crop_box is None:
            # Compute bounding box per frame, then take the union across all frames
            union_x1 = canvas_width
            union_y1 = canvas_height
            union_x2 = 0
            union_y2 = 0

            for m_pil in l_masks:
                _mask_img = mask2image(image2mask(m_pil))
                bluredmask = gaussian_blur(_mask_img, 20).convert('L')
                if detect == "min_bounding_rect":
                    (x, y, w, h) = min_bounding_rect(bluredmask)
                elif detect == "max_inscribed_rect":
                    (x, y, w, h) = max_inscribed_rect(bluredmask)
                else:
                    (x, y, w, h) = mask_area(_mask_img)

                if w > 0 and h > 0:
                    union_x1 = min(union_x1, x)
                    union_y1 = min(union_y1, y)
                    union_x2 = max(union_x2, x + w)
                    union_y2 = max(union_y2, y + h)

            # Fallback if no mask content found in any frame
            if union_x2 == 0 or union_y2 == 0:
                union_x1, union_y1, union_x2, union_y2 = 0, 0, canvas_width, canvas_height

            x1 = max(union_x1 - left_reserve, 0)
            y1 = max(union_y1 - top_reserve, 0)
            x2 = min(union_x2 + right_reserve, canvas_width)
            y2 = min(union_y2 + bottom_reserve, canvas_height)

            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                width = num_round_up_to_multiple(x2 - x1, multiple)
                height = num_round_up_to_multiple(y2 - y1, multiple)
                x1 = x1 - (width - (x2 - x1)) // 2
                y1 = y1 - (height - (y2 - y1)) // 2
                x2 = x1 + width
                y2 = y1 + height
            else:
                width = x2 - x1
                height = y2 - y1

            log(f"{self.NODE_NAME}: Union box across {len(l_masks)} frame(s). x={x1},y={y1},width={width},height={height}")
            crop_box = (x1, y1, x2, y2)

        # Draw preview using the first mask frame
        preview_image = l_masks[0].convert('RGB')
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)

        max_batch = max(len(l_images), len(l_masks))
        for i in range(max_batch):
            _canvas = tensor2pil(l_images[i] if i < len(l_images) else l_images[-1]).convert('RGB')
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            ret_images.append(pil2tensor(_canvas.crop(crop_box)))
            ret_masks.append(image2mask(_mask.crop(crop_box)))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), list(crop_box), pil2tensor(preview_image),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V2": CropByMaskV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V2": "LayerUtility: CropByMask V2"
}