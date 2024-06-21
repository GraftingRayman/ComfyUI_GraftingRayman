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

import folder_paths

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
