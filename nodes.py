import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

class GRPromptSelector:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "positive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a2": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "positive_a3": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a4": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a5": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a6": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "always_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "negative_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","STRING",)
    RETURN_NAMES = ("positive","negative","prompts",)
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman"
        


    def select_prompt(self, clip, positive_a1, positive_a2, positive_a3, positive_a4, positive_a5, positive_a6, always_a1, negative_a1, select_prompt):

        if select_prompt == 1:
            clipa = positive_a1
        elif select_prompt == 2:
            clipa = positive_a2
        elif select_prompt == 3:
            clipa = positive_a3
        elif select_prompt == 4:
            clipa = positive_a4
        elif select_prompt == 5:
            clipa = positive_a5
        elif select_prompt == 6:
            clipa = positive_a6
        positive = clipa + ", " + always_a1
        prompts = "positive:\n" + positive + "\n\nnegative: \n" + negative_a1
        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)
        return ([[condP, {"pooled_output": pooledP}]],[[condN, {"pooled_output": pooledP}]], prompts )



class GRImageResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "ImageProcessing"

    def resize_image(self, image, height, width):
        input_image = image.permute((0, 3, 1, 2))
        resized_image = TF.resize(input_image, (height, width))
        resized_image = resized_image.permute((0, 2, 3, 1))
        return (resized_image,)
        


class GRMaskResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize_mask"
    CATEGORY = "ImageProcessing"

    def resize_mask(self, mask, height, width):
        # Resize the mask tensor
        resized_mask = TF.resize(mask, (height, width))
        return resized_mask,



class GRMaskCreate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "transparent_width_percentage": ("FLOAT", {"min": 0, "max": 1}),
                "position_percentage": ("FLOAT", {"min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "ImageProcessing"

    def create_mask(self, height, width, transparent_width_percentage, position_percentage):
        # Calculate the width of the transparent area
        transparent_width = int(width * transparent_width_percentage)

        # Calculate the position of the transparent area
        position = int(width * position_percentage)

        # Create a blank mask tensor with the specified height and width
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)

        # Determine the starting x-coordinate for the transparent area based on the position
        x_start = max(0, min(width - transparent_width, position))

        # Fill the specified area with transparency
        mask[:, :, :, x_start:x_start + transparent_width] = 1.

        return mask










class GRMultiMaskCreate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "num_masks": ("INT", {"min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("MASK",) * 8
    RETURN_NAMES = tuple(f"MASK{i+1}" for i in range(8))
    FUNCTION = "create_multi_masks"
    CATEGORY = "ImageProcessing"

    def create_multi_masks(self, height, width, num_masks):
        masks = []

        for i in range(num_masks):
            # Calculate the width of the transparent area (example calculation)
            transparent_width_percentage = (i + 1) / 10
            transparent_width = int(width * transparent_width_percentage)

            # Calculate the position of the transparent area (example calculation)
            position_percentage = (i + 1) / 10
            position = int(width * position_percentage)

            # Create a blank mask image with the specified height and width
            mask = Image.new("L", (width, height), 0)

            # Calculate the coordinates for the transparent area
            x1 = max(0, min(width - transparent_width, position))
            x2 = min(width, x1 + transparent_width)

            # Fill the transparent area with white (255)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, 0, x2, height], fill=255)
            del draw

            # Convert the mask image to a tensor
            mask_np = np.array(mask, dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

            masks.append(mask_tensor)

        return tuple(masks)
