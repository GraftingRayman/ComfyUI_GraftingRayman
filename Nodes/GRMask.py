import random
import os
import io
import math
import numpy as np
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import latent_preview
from clip import tokenize, model
from PIL import Image, ImageOps, ImageSequence, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args
import random
import time
import cv2

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
    CATEGORY = "GraftingRayman/Mask"


    def create_mask(self, height, width, mask_size, seed):
        mask_dim = int(min(height, width) * mask_size)
        
        if mask_dim == 0:
            raise ValueError("mask_size is too small, resulting in zero mask dimensions.")
        
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        
        x_start = random.randint(0, width - mask_dim)
        y_start = random.randint(0, height - mask_dim)
        
        mask[:, :, y_start:y_start + mask_dim, x_start:x_start + mask_dim] = 1.
        
        return mask


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
    CATEGORY = "GraftingRayman/Mask"


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
    CATEGORY = "GraftingRayman/Mask"


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
    CATEGORY = "GraftingRayman/Mask"


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

class GRImageMask:
    _channels = ["alpha", "red", "green", "blue", "all"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "invert": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "blur_radius": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "blur_radius_expand": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "channel": (
                    cls._channels,
                    {"default": "all"},
                ),
                "expand": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "contract": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "GraftingRayman/Mask"


    def create_mask(self, image, threshold, invert, blur_radius, blur_radius_expand, brightness, contrast, channel, expand, contract):
        # Convert the image tensor to a PIL image
        image_np = image.cpu().numpy()
        pil_image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))

        # Select the specified channel
        if channel == "alpha":
            pil_image = pil_image.convert("RGBA")
            channel_image = pil_image.split()[-1]
        elif channel == "red":
            channel_image = pil_image.split()[0]
        elif channel == "green":
            channel_image = pil_image.split()[1]
        elif channel == "blue":
            channel_image = pil_image.split()[2]
        else:  # "all"
            channel_image = pil_image.convert("L")
        binary_mask = channel_image.point(lambda p: p > threshold * 255 and 255)
        if invert:
            binary_mask = ImageOps.invert(binary_mask)
        if blur_radius > 0.0:
            binary_mask = binary_mask.filter(ImageFilter.GaussianBlur(blur_radius))
        if brightness != 1.0:
            binary_mask = ImageEnhance.Brightness(binary_mask).enhance(brightness)
        if contrast != 1.0:
            binary_mask = ImageEnhance.Contrast(binary_mask).enhance(contrast)
        binary_mask_np = np.array(binary_mask)
        if expand > 0:
            kernel = np.ones((expand * 2 + 1, expand * 2 + 1), np.uint8)
            binary_mask_np = cv2.dilate(binary_mask_np, kernel, iterations=1)
        if contract > 0:
            kernel = np.ones((contract * 2 + 1, contract * 2 + 1), np.uint8)
            binary_mask_np = cv2.erode(binary_mask_np, kernel, iterations=1)
        binary_mask = Image.fromarray(binary_mask_np)
        if blur_radius_expand > 0.0:
            binary_mask = binary_mask.filter(ImageFilter.GaussianBlur(blur_radius_expand))
        mask_tensor = torch.tensor(np.array(binary_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        return mask_tensor



class GRMaskCreateRandomMulti:
    _available_colours = {
        "amethyst": "#9966CC", "black": "#000000", "blue": "#0000FF", "cyan": "#00FFFF",
        "diamond": "#B9F2FF", "emerald": "#50C878", "gold": "#FFD700", "gray": "#808080",
        "green": "#008000", "lime": "#00FF00", "magenta": "#FF00FF", "maroon": "#800000",
        "navy": "#000080", "neon_blue": "#1B03A3", "neon_green": "#39FF14", "neon_orange": "#FF6103",
        "neon_pink": "#FF10F0", "neon_yellow": "#DFFF00", "olive": "#808000", "platinum": "#E5E4E2",
        "purple": "#800080", "red": "#FF0000", "rose_gold": "#B76E79", "ruby": "#E0115F",
        "sapphire": "#0F52BA", "silver": "#C0C0C0", "teal": "#008080", "topaz": "#FFCC00",
        "white": "#FFFFFF", "yellow": "#FFFF00"
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": None}),
                "mask_size": ("FLOAT", {"min": 0.01, "max": 1, "step": 0.01}),
                "mask_number": ("INT", {"min": 1}),
                "exclude_borders": ("BOOLEAN", {"default": False}),  # Moved here
                "border_margin_height": ("INT", {"min": 0, "default": 15}),
                "border_margin_width": ("INT", {"min": 0, "default": 15}),
                "min_distance": ("INT", {"min": 0, "default": 10}),
                "ring_color": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "ring_type": (
                    ["solid", "dashed", "dotted"],
                    {"default": "solid"}
                ),
                "use_image_dimensions": ("BOOLEAN", {"default": False}),
                "height": ("INT", {"min": 1, "default": 256}),
                "width": ("INT", {"min": 1, "default": 256}),
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "create_masked_image_with_rings"
    CATEGORY = "GraftingRayman/Tiles"

    def create_masked_image_with_rings(self, image=None, seed=None, mask_size=None, mask_number=None, exclude_borders=False, ring_color="black", ring_type="solid", border_margin_height=15, border_margin_width=15, min_distance=10, use_image_dimensions=False, height=256, width=256):
        if image is None:
            image = torch.zeros((1, height, width, 3), dtype=torch.float32)

        batch_size, orig_height, orig_width, channels = image.size()

        if use_image_dimensions:
            height = orig_height
            width = orig_width

        if seed is not None:
            random.seed(seed)  # Set the seed if provided

        mask = self.create_mask(height, width, mask_size, mask_number, seed, exclude_borders, border_margin_height, border_margin_width, min_distance)
        grow_distance = 10  # Hardcoded grow distance
        grown_mask = self.grow_mask(mask, grow_distance)
        grown_mask2 = self.grow_mask(grown_mask, grow_distance)  # Grow grown_mask again

        ringed_image2 = self.apply_curved_rings_to_image(image.clone(), grown_mask, grown_mask2, ring_color, ring_type)  # Second ringed image with curved corners

        return (mask, ringed_image2)

    def create_mask(self, height, width, mask_size, mask_number, seed=0, exclude_borders=False, border_margin_height=15, border_margin_width=15, min_distance=10):
        if seed != 0:
            random.seed(seed)
        
        mask_dim = int(min(height, width) * mask_size)
        
        if mask_dim <= 0:
            raise ValueError("mask_size is too small, resulting in zero mask dimensions.")
        
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        
        border_margin_height = int(height * (border_margin_height / 100))
        border_margin_width = int(width * (border_margin_width / 100))
        min_distance = int(min(height, width) * (min_distance / 100))  # Minimum distance between masks

        placed_masks = []

        for _ in range(mask_number):
            placed = False
            attempts = 0

            while not placed and attempts < 100:  # Limit the number of attempts to avoid infinite loops
                if exclude_borders:
                    x_start = random.randint(border_margin_width, width - mask_dim - border_margin_width)
                    y_start = random.randint(border_margin_height, height - mask_dim - border_margin_height)
                else:
                    x_start = random.randint(0, width - mask_dim)
                    y_start = random.randint(0, height - mask_dim)
                
                # Check if the new mask is at least min_distance away from all existing masks
                too_close = False
                for mx, my in placed_masks:
                    if abs(mx - x_start) < mask_dim + min_distance and abs(my - y_start) < mask_dim + min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    mask[:, :, y_start:y_start + mask_dim, x_start:x_start + mask_dim] = 1.
                    placed_masks.append((x_start, y_start))
                    placed = True
                
                attempts += 1

            if not placed:
                raise ValueError("Could not place mask without overlap within the given number of attempts.")
        
        return mask

    def grow_mask(self, mask, distance):
        grown_mask = mask.clone()
        for _ in range(distance):
            grown_mask = torch.nn.functional.max_pool2d(grown_mask, kernel_size=3, stride=1, padding=1)
        return grown_mask

    def apply_curved_rings_to_image(self, image, mask, grown_mask, ring_color, ring_type):
        color_dict = {
            "amethyst": "#9966CC", "black": "#000000", "blue": "#0000FF", "cyan": "#00FFFF",
            "diamond": "#B9F2FF", "emerald": "#50C878", "gold": "#FFD700", "gray": "#808080",
            "green": "#008000", "lime": "#00FF00", "magenta": "#FF00FF", "maroon": "#800000",
            "navy": "#000080", "neon_blue": "#1B03A3", "neon_green": "#39FF14", "neon_orange": "#FF6103",
            "neon_pink": "#FF10F0", "neon_yellow": "#DFFF00", "olive": "#808000", "platinum": "#E5E4E2",
            "purple": "#800080", "red": "#FF0000", "rose_gold": "#B76E79", "ruby": "#E0115F",
            "sapphire": "#0F52BA", "silver": "#C0C0C0", "teal": "#008080", "topaz": "#FFCC00",
            "white": "#FFFFFF", "yellow": "#FFFF00"
        }
        
        hex_color = color_dict[ring_color]
        ring_color_rgb = torch.tensor([int(hex_color[i:i+2], 16) for i in (1, 3, 5)], dtype=image.dtype, device=image.device)

        print(f"Applying {ring_type} ring with color {ring_color} (RGB: {ring_color_rgb})")

        ring_mask = grown_mask - mask  # This creates the ring region

        # Apply Gaussian blur to smooth the edges
        ring_mask = F.conv2d(ring_mask, self.gaussian_kernel(5, 1.0), padding=2)

        # Threshold the blurred mask to create a smooth ring mask
        ring_mask = (ring_mask > 0.5).float()
        
        for batch in range(image.size(0)):  # Iterate over each image in the batch
            for y in range(ring_mask.size(2)):  # Iterate over the height of the mask
                for x in range(ring_mask.size(3)):  # Iterate over the width of the mask
                    if ring_mask[0, 0, y, x] > 0.5:  # Thresholding after blur
                        image[batch, y, x, :] = ring_color_rgb

        return image

    def gaussian_kernel(self, kernel_size: int, sigma: float):
        # Create a 2D Gaussian kernel
        grid = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        x_grid, y_grid = torch.meshgrid(grid, grid)
        kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel
