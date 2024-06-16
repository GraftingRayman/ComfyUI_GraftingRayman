import torch
import random

class GRMaskCreateRandom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "mask_size": ("FLOAT", {"min": 0.01, "max": 1, "step": 0.01}),
                "seed": ("INT", {"min": 1, "default": 0}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "GraftingRayman"

    def create_mask(self, height, width, mask_size, seed):
        mask_dim = int(min(height, width) * mask_size)
        
        if mask_dim == 0:
            raise ValueError("mask_size is too small, resulting in zero mask dimensions.")
        
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        
        x_start = random.randint(0, width - mask_dim)
        y_start = random.randint(0, height - mask_dim)
        
        mask[:, :, y_start:y_start + mask_dim, x_start:x_start + mask_dim] = 1.
        
        return mask
