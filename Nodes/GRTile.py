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
from PIL import Image, ImageOps, ImageSequence, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args
import random
import time

import folder_paths

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




class GRFlipTileInverted:
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

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "GRtile_image"
    CATEGORY = "GraftingRayman"

    def GRtile_image(self, image, rows, columns, colour, border=0):
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
        inverted_image = torch.zeros_like(new_image)

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
                    flipped_tile = torch.flip(tile, dims=[2])
                    new_image[:, y_start:y_end, x_start:x_end, :] = flipped_tile
                    inverted_tile = 1.0 - flipped_tile
                    inverted_image[:, y_start:y_end, x_start:x_end, :] = inverted_tile
                else:
                    new_image[:, y_start:y_end, x_start:x_end, :] = tile
                    inverted_image[:, y_start:y_end, x_start:x_end, :] = tile

        return (new_image, inverted_image)

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

class GRFlipTileRedRing:
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
                "border_thickness": ("INT", {"min": 1, "default": 5}),
                "seed": ("INT", {"default": None}),  # Adding seed input
            },
            "optional": {
                "flipped_tile_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "GRRedRingtile_image"
    CATEGORY = "GraftingRayman"

    def GRRedRingtile_image(self, image, rows, columns, colour, border=0, border_thickness=5, seed=None, flipped_tile_image=None):
        batch_size, orig_height, orig_width, channels = image.size()

        if seed is not None:
            random.seed(seed)  # Set the seed if provided

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
        red_bordered_image = torch.clone(new_image)

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
                    if flipped_tile_image is not None:
                        flipped_tile = flipped_tile_image
                    else:
                        flipped_tile = torch.flip(tile, dims=[2])
                    
                    new_image[:, y_start:y_end, x_start:x_end, :] = flipped_tile

                    red_bordered_image[:, y_start:y_end, x_start:x_end, :] = flipped_tile
                    red_border = self.get_colour_value("red")

                    # Apply red border to the flipped tile
                    red_bordered_image[:, y_start:y_start+border_thickness, x_start:x_end, :] = red_border
                    red_bordered_image[:, y_end-border_thickness:y_end, x_start:x_end, :] = red_border
                    red_bordered_image[:, y_start:y_end, x_start:x_start+border_thickness, :] = red_border
                    red_bordered_image[:, y_start:y_end, x_end-border_thickness:x_end, :] = red_border

                else:
                    new_image[:, y_start:y_end, x_start:x_end, :] = tile
                    red_bordered_image[:, y_start:y_end, x_start:x_end, :] = tile

        return (new_image, red_bordered_image)

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
