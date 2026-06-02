import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask


class RestoreCropBox:

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_data": ("CROP_DATA",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'restore_crop_box'
    CATEGORY = '😺dzNodes/LayerUtility'

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_data,
                         croped_mask=None):

        b_images = []
        l_images = []
        l_masks  = []
        ret_images = []
        ret_masks  = []

        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in croped_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))

        if croped_mask is not None:
            if croped_mask.dim() == 2:
                croped_mask = torch.unsqueeze(croped_mask, 0)
            l_masks = []
            for m in croped_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))

        for i in range(max_batch):
            bg_t      = b_images[i] if i < len(b_images) else b_images[-1]
            crop_t    = l_images[i] if i < len(l_images) else l_images[-1]
            mask_pil  = l_masks[i]  if i < len(l_masks)  else l_masks[-1]

            # Per-frame crop metadata — fall back to last entry if batch sizes differ
            data = crop_data[i] if i < len(crop_data) else crop_data[-1]
            x1, y1, x2, y2 = data["box"]
            scale    = data["scale"]
            off_x    = data["offset_x"]
            off_y    = data["offset_y"]
            cw       = data["content_w"]
            ch       = data["content_h"]

            _canvas   = tensor2pil(bg_t).convert('RGB')
            _layer    = tensor2pil(crop_t).convert('RGB')

            # --- Reverse letterbox ---
            # 1. Extract content region from the output frame (strip padding)
            content_img  = _layer.crop((off_x, off_y, off_x + cw, off_y + ch))
            content_mask = mask_pil.crop((off_x, off_y, off_x + cw, off_y + ch))

            # 2. Scale back to original crop size
            orig_w = x2 - x1
            orig_h = y2 - y1
            if orig_w > 0 and orig_h > 0:
                restored_img  = content_img.resize((orig_w, orig_h), Image.LANCZOS)
                restored_mask = content_mask.resize((orig_w, orig_h), Image.LANCZOS)
            else:
                restored_img  = content_img
                restored_mask = content_mask

            # 3. Paste back onto background at original position
            ret_mask_img = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(restored_img, box=(x1, y1), mask=restored_mask)
            ret_mask_img.paste(restored_mask, box=(x1, y1))

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(ret_mask_img))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: RestoreCropBox": RestoreCropBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RestoreCropBox": "LayerUtility: RestoreCropBox"
}
