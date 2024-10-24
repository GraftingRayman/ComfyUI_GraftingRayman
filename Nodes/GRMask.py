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

class GRMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "mask_width": ("INT", {"min": 1}),  # Mask width in pixels
                "mask_height": ("INT", {"min": 1}),  # Mask height in pixels
                "mask_position_v": (["bottom", "middle", "top"],),  # Vertical position options
                "mask_position_h": (["left", "center", "right"],),  # Horizontal position options
                "offset_x": ("INT", {"min": -1024, "max": 1024}),  # Offset for x-axis
                "offset_y": ("INT", {"min": -1024, "max": 1024}),  # Offset for y-axis
                "mask_shape": (["circle", "square", "rectangle", "oval"],),  # Mask shape options
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "GraftingRayman/Mask"

    def create_mask(self, height, width, mask_width, mask_height, mask_position_v, mask_position_h, offset_x, offset_y, mask_shape):
        # Initialize a blank mask
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)

        # Calculate mask position based on the vertical alignment
        if mask_position_v == "top":
            y_start = 0
        elif mask_position_v == "middle":
            y_start = (height - mask_height) // 2
        else:  # "bottom"
            y_start = height - mask_height

        # Calculate mask position based on the horizontal alignment
        if mask_position_h == "left":
            x_start = 0
        elif mask_position_h == "center":
            x_start = (width - mask_width) // 2
        else:  # "right"
            x_start = width - mask_width

        # Apply the offsets
        x_start = max(0, min(width - mask_width, x_start + offset_x))
        y_start = max(0, min(height - mask_height, y_start + offset_y))

        # Create the mask based on the selected shape
        if mask_shape == "rectangle" or mask_shape == "square":
            mask[:, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1.
        elif mask_shape == "circle":
            # Draw a circular mask (simplified for the sake of demonstration)
            radius = min(mask_width, mask_height) // 2
            center_x = x_start + mask_width // 2
            center_y = y_start + mask_height // 2
            for i in range(height):
                for j in range(width):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 <= radius ** 2:
                        mask[:, :, i, j] = 1.
        elif mask_shape == "oval":
            # Draw an oval mask
            center_x = x_start + mask_width // 2
            center_y = y_start + mask_height // 2
            for i in range(height):
                for j in range(width):
                    if ((i - center_y) ** 2) / (mask_height // 2) ** 2 + ((j - center_x) ** 2) / (mask_width // 2) ** 2 <= 1:
                        mask[:, :, i, j] = 1.

        return mask

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
                "seed": ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1}),
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

    _available_shapes = [
        "rectangular", "circular", "elliptical", "triangular", 
        "polygonal", "star", "random", "hexagon", "pentagon"
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1}),
                "mask_size": ("FLOAT", {"min": 0.01, "max": 1, "step": 0.01}),
                "mask_number": ("INT", {"min": 1}),
                "exclude_borders": ("BOOLEAN", {"default": False}),
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
                "distance_from_boundary": ("INT", {"min": 1, "default": 10}),
                "ring_thickness": ("INT", {"min": 1, "default": 5}),
                "mask_shape": (
                    cls._available_shapes,
                    {"default": "rectangular"}
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

    def create_masked_image_with_rings(self, image=None, seed=None, mask_size=None, mask_number=None, exclude_borders=False, ring_color="black", ring_type="solid", border_margin_height=15, border_margin_width=15, min_distance=10, distance_from_boundary=10, ring_thickness=5, mask_shape="rectangular", use_image_dimensions=False, height=256, width=256):
        if image is None:
            image = torch.zeros((1, height, width, 3), dtype=torch.float32)

        batch_size, orig_height, orig_width, channels = image.size()

        if use_image_dimensions:
            height = orig_height
            width = orig_width

        if seed is not None:
            random.seed(seed)  # Set the seed if provided

        mask = self.create_mask(height, width, mask_size, mask_number, seed, exclude_borders, border_margin_height, border_margin_width, min_distance, mask_shape)
        grown_mask_outer = self.grow_mask(mask, distance_from_boundary + ring_thickness)
        grown_mask_inner = self.grow_mask(mask, distance_from_boundary)

        ring_mask = grown_mask_outer - grown_mask_inner

        ringed_image = self.apply_rings_to_image(image.clone(), ring_mask, ring_color, ring_type)

        return (mask, ringed_image)

    def create_mask(self, height, width, mask_size, mask_number, seed=0, exclude_borders=False, border_margin_height=15, border_margin_width=15, min_distance=10, mask_shape="rectangular"):
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
                    actual_shape = mask_shape if mask_shape != "random" else random.choice(self._available_shapes[:-1])
                    if actual_shape == "rectangular":
                        mask[:, :, y_start:y_start + mask_dim, x_start:x_start + mask_dim] = 1.
                    elif actual_shape == "circular":
                        self.draw_circle(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "elliptical":
                        self.draw_ellipse(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "triangular":
                        self.draw_triangle(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "polygonal":
                        self.draw_polygon(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "star":
                        self.draw_star(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "hexagon":
                        self.draw_hexagon(mask, x_start, y_start, mask_dim)
                    elif actual_shape == "pentagon":
                        self.draw_pentagon(mask, x_start, y_start, mask_dim)
                    placed_masks.append((x_start, y_start))
                    placed = True
                
                attempts += 1

            if not placed:
                raise ValueError("Could not place mask without overlap within the given number of attempts.")
        
        return mask

    def draw_circle(self, mask, x_center, y_center, diameter):
        radius = diameter // 2
        for y in range(-radius, radius):
            for x in range(-radius, radius):
                if x**2 + y**2 <= radius**2:
                    if 0 <= x_center + x < mask.shape[3] and 0 <= y_center + y < mask.shape[2]:
                        mask[0, 0, y_center + y, x_center + x] = 1.

    def draw_ellipse(self, mask, x_center, y_center, diameter):
        a = diameter // 2
        b = diameter // 3  # Aspect ratio of ellipse
        for y in range(-b, b):
            for x in range(-a, a):
                if (x**2) / (a**2) + (y**2) / (b**2) <= 1:
                    if 0 <= x_center + x < mask.shape[3] and 0 <= y_center + y < mask.shape[2]:
                        mask[0, 0, y_center + y, x_center + x] = 1.

    def draw_triangle(self, mask, x_start, y_start, side_length):
        height = int(math.sqrt(side_length**2 - (side_length / 2)**2))
        for y in range(height):
            x_min = x_start - (y * side_length) // height
            x_max = x_start + (y * side_length) // height
            mask[0, 0, y_start + y, max(0, x_min):min(mask.shape[3], x_max)] = 1.

    def draw_polygon(self, mask, x_center, y_center, diameter, sides=6):
        radius = diameter // 2
        angle = 2 * math.pi / sides
        points = [(x_center + radius * math.cos(i * angle), y_center + radius * math.sin(i * angle)) for i in range(sides)]
        self.draw_filled_polygon(mask, points)

    def draw_star(self, mask, x_center, y_center, diameter):
        radius = diameter // 2
        angle = math.pi / 5
        points = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius // 2
            points.append((x_center + r * math.cos(i * angle), y_center + r * math.sin(i * angle)))
        self.draw_filled_polygon(mask, points)

    def draw_hexagon(self, mask, x_center, y_center, diameter):
        self.draw_polygon(mask, x_center, y_center, diameter, sides=6)

    def draw_pentagon(self, mask, x_center, y_center, diameter):
        self.draw_polygon(mask, x_center, y_center, diameter, sides=5)

    def draw_filled_polygon(self, mask, points):
        def is_point_in_polygon(x, y, poly):
            num = len(poly)
            j = num - 1
            c = False
            for i in range(num):
                if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                        (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
                    c = not c
                j = i
            return c

        for y in range(mask.shape[2]):
            for x in range(mask.shape[3]):
                if is_point_in_polygon(x, y, points):
                    mask[0, 0, y, x] = 1

    def grow_mask(self, mask, distance):
        grown_mask = mask.clone()
        for _ in range(distance):
            grown_mask = torch.nn.functional.max_pool2d(grown_mask, kernel_size=3, stride=1, padding=1)
        return grown_mask

    def apply_rings_to_image(self, image, ring_mask, ring_color, ring_type):
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
