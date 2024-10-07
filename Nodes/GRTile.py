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
from comfy.utils import ProgressBar

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
    CATEGORY = "GraftingRayman/Tiles"

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
    CATEGORY = "GraftingRayman/Tiles"


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
    CATEGORY = "GraftingRayman/Tiles"


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
                "seed": ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1}),
            },
            "optional": {
                "flipped_tile_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "GRRedRingtile_image"
    CATEGORY = "GraftingRayman/Tiles"


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

class GRCheckeredBoard:
    def __init__(self):
        self.colors = {
            "black": torch.tensor([0, 0, 0], dtype=torch.float32) / 255.0,
            "blue": torch.tensor([0, 0, 255], dtype=torch.float32) / 255.0,
            "brown": torch.tensor([165, 42, 42], dtype=torch.float32) / 255.0,
            "cyan": torch.tensor([0, 255, 255], dtype=torch.float32) / 255.0,
            "darkblue": torch.tensor([0, 0, 139], dtype=torch.float32) / 255.0,
            "darkgray": torch.tensor([169, 169, 169], dtype=torch.float32) / 255.0,
            "darkgreen": torch.tensor([0, 100, 0], dtype=torch.float32) / 255.0,
            "darkred": torch.tensor([139, 0, 0], dtype=torch.float32) / 255.0,
            "gray": torch.tensor([128, 128, 128], dtype=torch.float32) / 255.0,
            "green": torch.tensor([0, 255, 0], dtype=torch.float32) / 255.0,
            "lightblue": torch.tensor([173, 216, 230], dtype=torch.float32) / 255.0,
            "lightgray": torch.tensor([211, 211, 211], dtype=torch.float32) / 255.0,
            "lightgreen": torch.tensor([144, 238, 144], dtype=torch.float32) / 255.0,
            "lightred": torch.tensor([255, 182, 193], dtype=torch.float32) / 255.0,
            "magenta": torch.tensor([255, 0, 255], dtype=torch.float32) / 255.0,
            "maroon": torch.tensor([128, 0, 0], dtype=torch.float32) / 255.0,
            "navy": torch.tensor([0, 0, 128], dtype=torch.float32) / 255.0,
            "olive": torch.tensor([128, 128, 0], dtype=torch.float32) / 255.0,
            "orange": torch.tensor([255, 165, 0], dtype=torch.float32) / 255.0,
            "pink": torch.tensor([255, 192, 203], dtype=torch.float32) / 255.0,
            "purple": torch.tensor([128, 0, 128], dtype=torch.float32) / 255.0,
            "red": torch.tensor([255, 0, 0], dtype=torch.float32) / 255.0,
            "salmon": torch.tensor([250, 128, 114], dtype=torch.float32) / 255.0,
            "silver": torch.tensor([192, 192, 192], dtype=torch.float32) / 255.0,
            "teal": torch.tensor([0, 128, 128], dtype=torch.float32) / 255.0,
            "violet": torch.tensor([238, 130, 238], dtype=torch.float32) / 255.0,
            "white": torch.tensor([255, 255, 255], dtype=torch.float32) / 255.0,
            "yellow": torch.tensor([255, 255, 0], dtype=torch.float32) / 255.0,
            "gold": torch.tensor([255, 215, 0], dtype=torch.float32) / 255.0,
            "khaki": torch.tensor([240, 230, 140], dtype=torch.float32) / 255.0,
            "lime": torch.tensor([50, 205, 50], dtype=torch.float32) / 255.0,
            "turquoise": torch.tensor([64, 224, 208], dtype=torch.float32) / 255.0,
        }

    @classmethod
    def INPUT_TYPES(cls):
        colors = list(cls().colors.keys())
        return {
            "required": {
                "rows": ("INT", {"min": 1}),
                "columns": ("INT", {"min": 1}),
                "tile_size": ("INT", {"min": 1}),
                "color1": (colors,),
                "color2": (colors,),
                "border": ("INT", {"min": 0, "default": 0}),
                "border_color": (colors,),
                "outer_border": ("INT", {"min": 0, "default": 0}),
                "outer_border_color": (colors,)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "create_checkered_board"
    CATEGORY = "GraftingRayman/Patterns"

    def create_checkered_board(self, rows, columns, tile_size, color1, color2, border=0, border_color="black", outer_border=0, outer_border_color="black"):
        total_tile_size = tile_size + 2 * border
        board_height = rows * total_tile_size
        board_width = columns * total_tile_size
        outer_board_height = board_height + 2 * outer_border
        outer_board_width = board_width + 2 * outer_border

        image = torch.zeros((1, outer_board_height, outer_board_width, 3), dtype=torch.float32)
        mask = torch.ones((1, outer_board_height, outer_board_width), dtype=torch.float32)  # Start with a mask of ones (opaque)
        
        color1_value = self.colors[color1]
        color2_value = self.colors[color2]
        border_color_value = self.colors[border_color]
        outer_border_color_value = self.colors[outer_border_color]

        # Determine the lightest color for the mask
        if color1_value.mean() > color2_value.mean():
            transparent_color_value = color1_value
        else:
            transparent_color_value = color2_value

        # Fill the outer border
        if outer_border > 0:
            image[:, :outer_border, :, :] = outer_border_color_value  # Top border
            image[:, -outer_border:, :, :] = outer_border_color_value  # Bottom border
            image[:, :, :outer_border, :] = outer_border_color_value  # Left border
            image[:, :, -outer_border:, :] = outer_border_color_value  # Right border

        batch = rows * columns
        pbar = ProgressBar(batch)
        
        for i in range(rows):
            for j in range(columns):
                idx = i * columns + j  # Calculate the current index for progress bar

                y_start = outer_border + i * total_tile_size
                y_end = y_start + total_tile_size
                x_start = outer_border + j * total_tile_size
                x_end = x_start + total_tile_size

                if (i + j) % 2 == 0:
                    tile_color_value = color1_value
                else:
                    tile_color_value = color2_value

                image[:, y_start + border:y_end - border, x_start + border:x_end - border, :] = tile_color_value
                
                # Update mask for the transparent color
                if torch.equal(tile_color_value, transparent_color_value):
                    mask[:, y_start + border:y_end - border, x_start + border:x_end - border] = 0  # Set transparent

                if border > 0:
                    image[:, y_start:y_start + border, x_start:x_end, :] = border_color_value  # Top border
                    image[:, y_end - border:y_end, x_start:x_end, :] = border_color_value  # Bottom border
                    image[:, y_start:y_end, x_start:x_start + border, :] = border_color_value  # Left border
                    image[:, y_start:y_end, x_end - border:x_end, :] = border_color_value  # Right border
                
                pbar.update_absolute(idx)

        return image, mask

    def get_color_value(self, color):
        return self.colors[color]
