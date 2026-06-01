import torch
from .imagefunc import log


class MaskToValue:

    def __init__(self):
        self.NODE_NAME = 'MaskToValue'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_val": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.01}),
                "max_val": ("FLOAT", {"default": 48.0, "min": -9999.0, "max": 9999.0, "step": 0.01}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Pixels above this value count as white. Set to 0 for soft weighted average."}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("values_float", "values_int")
    FUNCTION = "mask_to_value"
    CATEGORY = "😺dzNodes/LayerUtility"

    def mask_to_value(self, mask, min_val, max_val, threshold=0.5):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        results_float = []
        for i in range(mask.shape[0]):
            frame = mask[i]
            if threshold > 0.0:
                coverage = (frame > threshold).float().mean().item()
            else:
                coverage = frame.mean().item()
            value = min_val + (max_val - min_val) * coverage
            value = max(min(value, max(min_val, max_val)), min(min_val, max_val))
            results_float.append(value)

        if len(results_float) == 1:
            out_float = results_float[0]
            out_int = int(round(out_float))
        else:
            out_float = results_float
            out_int = [int(round(v)) for v in results_float]

        log(f"{self.NODE_NAME} Processed {mask.shape[0]} frame(s).", message_type='finish')
        return (out_float, out_int)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: MaskToValue": MaskToValue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: MaskToValue": "LayerUtility: Mask To Value"
}
