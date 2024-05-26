import random
import os
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import latent_preview
from clip import tokenize, model
from PIL import Image, ImageDraw

class GRPromptSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        clip_type = ("CLIP",)
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True})
        return {"required": {
            "clip": clip_type,
            **{f"positive_a{i}": string_type for i in range(1, 7)},
            "always_a1": string_type,
            "negative_a1": string_type,
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompts")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman"

    def select_prompt(self, clip, **kwargs):
        select_prompt = kwargs["select_prompt"]
        positive_clip = kwargs[f"positive_a{select_prompt}"]
        always_a1 = kwargs["always_a1"]
        negative_a1 = kwargs["negative_a1"]

        positive = f"{positive_clip}, {always_a1}"
        prompts = f"positive:\n{positive}\n\nnegative:\n{negative_a1}"

        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)

        return ([[condP, {"pooled_output": pooledP}]], [[condN, {"pooled_output": pooledN}]], prompts)


class GRPromptSelectorMulti:
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
            "alwayspositive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "negative_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            }}


    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive1","positive2","positive3","positive4","positive5","positive6","negative",)
    FUNCTION = "select_promptmulti"
    CATEGORY = "GraftingRayman"
        


    def select_promptmulti(self, clip, positive_a1, positive_a2, positive_a3, positive_a4, positive_a5, positive_a6, alwayspositive_a1, negative_a1):

        positive1 = positive_a1 + ", " + alwayspositive_a1
        positive2 = positive_a2 + ", " + alwayspositive_a1
        positive3 = positive_a3 + ", " + alwayspositive_a1
        positive4 = positive_a4 + ", " + alwayspositive_a1
        positive5 = positive_a5 + ", " + alwayspositive_a1
        positive6 = positive_a6 + ", " + alwayspositive_a1
        tokensP1 = clip.tokenize(positive1)
        tokensP2 = clip.tokenize(positive2)
        tokensP3 = clip.tokenize(positive3)
        tokensP4 = clip.tokenize(positive4)
        tokensP5 = clip.tokenize(positive5)
        tokensP6 = clip.tokenize(positive6)
        tokensN1 = clip.tokenize(negative_a1)
        condP1, pooledP1 = clip.encode_from_tokens(tokensP1, return_pooled=True)
        condP2, pooledP2 = clip.encode_from_tokens(tokensP2, return_pooled=True)
        condP3, pooledP3 = clip.encode_from_tokens(tokensP3, return_pooled=True)
        condP4, pooledP4 = clip.encode_from_tokens(tokensP4, return_pooled=True)
        condP5, pooledP5 = clip.encode_from_tokens(tokensP5, return_pooled=True)
        condP6, pooledP6 = clip.encode_from_tokens(tokensP6, return_pooled=True)
        condN1, pooledN1 = clip.encode_from_tokens(tokensN1, return_pooled=True)

        return ([[condP1, {"pooled_output": pooledP1}]],[[condP2, {"pooled_output": pooledP2}]],[[condP3, {"pooled_output": pooledP3}]],[[condP4, {"pooled_output": pooledP4}]],[[condP5, {"pooled_output": pooledP5}]],[[condP6, {"pooled_output": pooledP6}]],[[condN1, {"pooled_output": pooledN1}]],)


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
    CATEGORY = "GraftingRayman"

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
    CATEGORY = "GraftingRayman"

    def resize_mask(self, mask, height, width):
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
                "mask_width": ("FLOAT", {"min": 0, "max": 1}),
                "position_percentage": ("FLOAT", {"min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "GraftingRayman"

    def create_mask(self, height, width, mask_width, position_percentage):
        transparent_width = int(width * mask_width)
        position = int(width * position_percentage)
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        x_start = max(0, min(width - transparent_width, position))
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
                "num_masks": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("MASK",) * 8  
    RETURN_NAMES = ("mask1", "mask2", "mask3", "mask4", "mask5", "mask6", "mask7", "mask8")
    FUNCTION = "create_masks"
    CATEGORY = "GraftingRayman"

    def create_masks(self, height, width, num_masks):
        masks = []
        transparent_width = width // num_masks
        for i in range(num_masks):
            mask = torch.zeros((1, height, width), dtype=torch.float32)
            start_x = i * transparent_width
            end_x = (i + 1) * transparent_width if i < num_masks - 1 else width
            mask[:, :, start_x:end_x] = 1.
            masks.append(mask)
        return tuple(masks)


class GRImageSize:
    def __init__(self):
        pass
                
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "height": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
            "width": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
            "standard": (["custom", "(SD) 512x512","(SDXL) 1024x1024","640x480 (VGA)", "800x600 (SVGA)", "960x544 (Half HD)","1024x768 (XGA)", "1280x720 (HD)", "1366x768 (HD)","1600x900 (HD+)","1920x1080 (Full HD or 1080p)","2560x1440 (Quad HD or 1440p)","3840x2160 (Ultra HD, 4K, or 2160p)","5120x2880 (5K)","7680x4320 (8K)"],),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },}

    RETURN_TYPES = ("INT","INT","LATENT")
    RETURN_NAMES = ("height","width","samples")
    FUNCTION = "image_size"
    CATEGORY = "GraftingRayman"
        


    def image_size(self, height, width, standard, batch_size=1):
        if standard == "custom":
            height = height
            width = width
        elif standard == "(SD) 512x512":
            width = 512
            height = 512
        elif standard == "(SDXL) 1024x1024":
            width = 1024
            height = 1024
        elif standard == "640x480 (VGA)":
            width = 640
            height = 480
        elif standard == "800x600 (SVGA)":
            width = 800
            height = 608
        elif standard == "960x544 (Half HD)":
            width = 960
            height = 544
        elif standard == "1024x768 (XGA)":
            width = 1024
            height = 768
        elif standard == "1280x720 (HD)":
            width = 1280
            height = 720
        elif standard == "1366x768 (HD)":
            width = 1360
            height = 768
        elif standard == "1600x900 (HD+)":
            width = 1600
            height = 896
        elif standard == "1920x1080 (Full HD or 1080p)":
            width = 1920
            height = 1088
        elif standard == "2560x1440 (Quad HD or 1440p)":
            width = 2560
            height = 1440
        elif standard == "3840x2160 (Ultra HD, 4K, or 2160p)":
            width = 3840
            height = 2160
        elif standard == "5120x2880 (5K)":
            width = 5120
            height = 2880
        elif standard == "7680x4320 (8K)":
            width = 7680
            height = 4320            
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
    
        return (height,width,{"samples":latent},)

class GRTileImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        popular_colors = [
            "black", "white", "red", "blue", "green", "purple", "yellow"
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"min": 1}),
                "rows": ("INT", {"min": 1}),
                "border": ("INT", {"min": 0, "default": 0}),
                "colour": (popular_colors,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tile_image"
    CATEGORY = "GraftingRayman"

    def tile_image(self, image, rows, columns, colour, border=0):
        batch_size, orig_height, orig_width, channels = image.size()
        
        if border > 0:
            orig_height_with_border = orig_height + 2 * border
            orig_width_with_border = orig_width + 2 * border
            
            bordered_image = torch.ones((batch_size, orig_height_with_border, orig_width_with_border, channels), dtype=image.dtype, device=image.device)
            bordered_image[:, border:-border, border:-border, :] = image
            border_color = self.get_colour_value(colour)
            bordered_image[:, :border, :, :] = border_color  # Top border
            bordered_image[:, -border:, :, :] = border_color  # Bottom border
            bordered_image[:, :, :border, :] = border_color  # Left border
            bordered_image[:, :, -border:, :] = border_color  # Right border
        else:
            bordered_image = image

        new_height = rows * orig_height_with_border if border > 0 else rows * orig_height
        new_width = columns * orig_width_with_border if border > 0 else columns * orig_width
        new_image = torch.zeros((batch_size, new_height, new_width, channels), dtype=image.dtype, device=image.device)

        num_tiles_height = (new_height + orig_height_with_border - 1) // orig_height
        num_tiles_width = (new_width + orig_width_with_border - 1) // orig_width

        for i in range(num_tiles_height):
            for j in range(num_tiles_width):
                y_start = i * orig_height_with_border
                y_end = min(y_start + orig_height_with_border, new_height)
                x_start = j * orig_width_with_border
                x_end = min(x_start + orig_width_with_border, new_width)
                new_image[:, y_start:y_end, x_start:x_end, :] = bordered_image[:, :y_end - y_start, :x_end - x_start, :]

        return (new_image,)

    def get_colour_value(self, colour):
        # Map color names to RGB values
        color_map = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "purple": [128, 0, 128],
            "yellow": [255, 255, 0],

        }
        return torch.tensor(color_map[colour], dtype=torch.float32)

class GRTileFlipImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        popular_colors = [
            "black", "white", "red", "blue", "green", "purple", "yellow"
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"min": 1}),
                "rows": ("INT", {"min": 1}),
                "border": ("INT", {"min": 0, "default": 0}),
                "colour": (popular_colors,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tile_image"
    CATEGORY = "GraftingRayman"

    def tile_image(self, image, rows, columns, colour, border=0):
        batch_size, orig_height, orig_width, channels = image.size()

        if border > 0:
            orig_height_with_border = orig_height + 2 * border
            orig_width_with_border = orig_width + 2 * border

            bordered_image = torch.ones((batch_size, orig_height_with_border, orig_width_with_border, channels), dtype=image.dtype, device=image.device)
            bordered_image[:, border:-border, border:-border, :] = image
            border_color = self.get_colour_value(colour)
            bordered_image[:, :border, :, :] = border_color  # Top border
            bordered_image[:, -border:, :, :] = border_color  # Bottom border
            bordered_image[:, :, :border, :] = border_color  # Left border
            bordered_image[:, :, -border:, :] = border_color  # Right border
        else:
            bordered_image = image

        new_height = rows * orig_height_with_border if border > 0 else rows * orig_height
        new_width = columns * orig_width_with_border if border > 0 else columns * orig_width
        new_image = torch.zeros((batch_size, new_height, new_width, channels), dtype=image.dtype, device=image.device)

        num_tiles_height = (new_height + orig_height_with_border - 1) // orig_height_with_border
        num_tiles_width = (new_width + orig_width_with_border - 1) // orig_width_with_border

        flip_tile = (random.randint(0, rows - 1), random.randint(0, columns - 1))

        for i in range(rows):
            for j in range(columns):
                y_start = i * orig_height_with_border
                y_end = min(y_start + orig_height_with_border, new_height)
                x_start = j * orig_width_with_border
                x_end = min(x_start + orig_width_with_border, new_width)

                tile = bordered_image[:, :y_end - y_start, :x_end - x_start, :]

                if (i, j) == flip_tile:
                    tile = torch.flip(tile, dims=[2])

                new_image[:, y_start:y_end, x_start:x_end, :] = tile

        return (new_image,)

    def get_colour_value(self, colour):
        # Map color names to RGB values
        color_map = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "purple": [128, 0, 128],
            "yellow": [255, 255, 0],
        }
        return torch.tensor(color_map[colour], dtype=torch.float32)

class GRMaskCreateRandom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "mask_size": ("FLOAT", {"min": 0, "max": 1}),
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

class GRStackImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        popular_colors = [
            "black", "white", "red", "blue", "green", "purple", "yellow"
        ]
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "border": ("INT", {"min": 0, "default": 0}),
                "colour": (popular_colors,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stack_images"
    CATEGORY = "GraftingRayman"

    def stack_images(self, image1, image2, colour, border=0):
        batch_size1, orig_height1, orig_width1, channels1 = image1.size()
        batch_size2, orig_height2, orig_width2, channels2 = image2.size()

        if batch_size1 != batch_size2 or channels1 != channels2 or orig_width1 != orig_width2:
            raise ValueError("Images must have the same batch size, width, and number of channels.")
        
        if border > 0:
            orig_height1_with_border = orig_height1 + 2 * border
            orig_height2_with_border = orig_height2 + 2 * border
            orig_width_with_border = orig_width1 + 2 * border
            
            bordered_image1 = torch.ones((batch_size1, orig_height1_with_border, orig_width_with_border, channels1), dtype=image1.dtype, device=image1.device)
            bordered_image2 = torch.ones((batch_size2, orig_height2_with_border, orig_width_with_border, channels2), dtype=image2.dtype, device=image2.device)

            bordered_image1[:, border:-border, border:-border, :] = image1
            bordered_image2[:, border:-border, border:-border, :] = image2

            border_color = self.get_colour_value(colour)
            bordered_image1[:, :border, :, :] = border_color  # Top border
            bordered_image1[:, -border:, :, :] = border_color  # Bottom border
            bordered_image1[:, :, :border, :] = border_color  # Left border
            bordered_image1[:, :, -border:, :] = border_color  # Right border

            bordered_image2[:, :border, :, :] = border_color  # Top border
            bordered_image2[:, -border:, :, :] = border_color  # Bottom border
            bordered_image2[:, :, :border, :] = border_color  # Left border
            bordered_image2[:, :, -border:, :] = border_color  # Right border
        else:
            bordered_image1 = image1
            bordered_image2 = image2

        new_height = (orig_height1_with_border if border > 0 else orig_height1) + (orig_height2_with_border if border > 0 else orig_height2)
        new_width = orig_width_with_border if border > 0 else orig_width1
        new_image = torch.zeros((batch_size1, new_height, new_width, channels1), dtype=image1.dtype, device=image1.device)

        new_image[:, :bordered_image1.size(1), :, :] = bordered_image1
        new_image[:, bordered_image1.size(1):, :, :] = bordered_image2

        return (new_image,)

    def get_colour_value(self, colour):
        # Map color names to RGB values
        color_map = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "purple": [128, 0, 128],
            "yellow": [255, 255, 0],
        }
        return torch.tensor(color_map[colour], dtype=torch.float32)
