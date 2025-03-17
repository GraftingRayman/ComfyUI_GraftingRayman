import torch

class GRImageMultiplier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiplier": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "step": 1
                }),
                "interleave": ("BOOLEAN", {"default": False}),
                "random_order": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": 0,
                    "min": -0x8000000000000000,
                    "max": 0x7FFFFFFFFFFFFFFF,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "multiply"
    CATEGORY = "GraftingRayman/Image"

    def multiply(self, images, multiplier, interleave, random_order, seed):
        if random_order:
            # Create sequential copies first
            multiplied = images.repeat_interleave(multiplier, dim=0)
            n = multiplied.size(0)
            
            # Create generator with seed
            generator = torch.Generator(device='cpu')
            if seed != 0:
                generator.manual_seed(seed)
            
            # Generate permutation and shuffle
            perm = torch.randperm(n, generator=generator)
            result = multiplied[perm.to(multiplied.device)]
        else:
            if interleave:
                # Interleaved repetition
                result = images.repeat(multiplier, 1, 1, 1)
            else:
                # Sequential repetition
                result = images.repeat_interleave(multiplier, dim=0)
        
        return (result,)

