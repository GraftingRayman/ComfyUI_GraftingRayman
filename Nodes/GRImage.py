import random
import os
import io
import math
import numpy as np
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T

import matplotlib.pyplot as plt
import latent_preview
from clip import tokenize, model
from PIL import Image, ImageOps, ImageSequence, ImageDraw, ImageFont, ImageChops
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args
import random
import time
import hashlib 
import folder_paths
import cv2
from torchvision import transforms as T
from rembg import remove as rembg_remove
from rembg.session_factory import new_session


class GRImageSize:
    _available_colours = {
        "amethyst": "#9966CC", "black": "#000000", "blue": "#0000FF", "cyan": "#00FFFF", "diamond": "#B9F2FF",
        "emerald": "#50C878", "gold": "#FFD700", "gray": "#808080", "green": "#008000", "lime": "#00FF00",
        "magenta": "#FF00FF", "maroon": "#800000", "navy": "#000080", "neon_blue": "#1B03A3", "neon_green": "#39FF14",
        "neon_orange": "#FF6103", "neon_pink": "#FF10F0", "neon_yellow": "#DFFF00", "olive": "#808000", "platinum": "#E5E4E2",
        "purple": "#800080", "red": "#FF0000", "rose_gold": "#B76E79", "ruby": "#E0115F", "sapphire": "#0F52BA",
        "silver": "#C0C0C0", "teal": "#008080", "topaz": "#FFCC00", "white": "#FFFFFF", "yellow": "#FFFF00"
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
                "width": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
                "standard": ([
                    "custom", "(SD) 512x512", "(SD2) 768x768", "(SD2) 768x512", "(SD2) 512x768 (Portrait)", 
                    "(SDXL) 1024x1024", "640x480 (VGA)", "800x600 (SVGA)", "960x544 (Half HD)", "1024x768 (XGA)", 
                    "1280x720 (HD)", "1366x768 (HD)", "1600x900 (HD+)", "1920x1080 (Full HD or 1080p)", 
                    "2560x1440 (Quad HD or 1440p)", "3840x2160 (Ultra HD, 4K, or 2160p)", "5120x2880 (5K)", 
                    "7680x4320 (8K)", "480x640 (VGA, Portrait)", "600x800 (SVGA, Portrait)", 
                    "544x960 (Half HD, Portrait)", "768x1024 (XGA, Portrait)", "720x1280 (HD, Portrait)", 
                    "768x1366 (HD, Portrait)", "900x1600 (HD+, Portrait)", "1080x1920 (Full HD or 1080p, Portrait)", 
                    "1440x2560 (Quad HD or 1440p, Portrait)", "2160x3840 (Ultra HD, 4K, or 2160p, Portrait)", 
                    "2880x5120 (5K, Portrait)", "4320x7680 (8K, Portrait)"
                ],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1}),
                "color": (list(sorted(cls._available_colours.keys())), {"default": "white"})
            },
            "optional": {
                "dimensions": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "LATENT", "INT", "IMAGE")
    RETURN_NAMES = ("height", "width", "batch_size", "samples", "seed", "empty_image")
    FUNCTION = "image_size"
    CATEGORY = "GraftingRayman/Images"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = hex_color * 2
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

    def generate_empty_image(self, width, height, batch_size, color):
        """Generate an empty image filled with the specified color."""
        color_value = int(self._available_colours[color].lstrip("#"), 16)
        r = torch.full([batch_size, height, width, 1], ((color_value >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color_value >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color_value) & 0xFF) / 0xFF)
        return torch.cat((r, g, b), dim=-1)

    def image_size(self, height, width, standard, batch_size=1, seed=None, color="white", dimensions=None):
        if dimensions is not None:
            standard = "custom"
            height, width, channels = dimensions.shape[-3:]

        if standard == "custom":
            height = height
            width = width
        else:
            standard_sizes = {
                "(SD) 512x512": (512, 512), "(SD2) 768x768": (768, 768), "(SD2) 768x512": (512, 768),
                "(SD2) 512x768 (Portrait)": (768, 512), "(SDXL) 1024x1024": (1024, 1024), "640x480 (VGA)": (480, 640),
                "800x600 (SVGA)": (600, 800), "960x544 (Half HD)": (544, 960), "1024x768 (XGA)": (768, 1024),
                "1280x720 (HD)": (720, 1280), "1366x768 (HD)": (768, 1366), "1600x900 (HD+)": (900, 1600),
                "1920x1080 (Full HD or 1080p)": (1080, 1920), "2560x1440 (Quad HD or 1440p)": (1440, 2560),
                "3840x2160 (Ultra HD, 4K, or 2160p)": (2160, 3840), "5120x2880 (5K)": (2880, 5120),
                "7680x4320 (8K)": (4320, 7680), "480x640 (VGA, Portrait)": (640, 480), "600x800 (SVGA, Portrait)": (800, 600),
                "544x960 (Half HD, Portrait)": (960, 544), "768x1024 (XGA, Portrait)": (1024, 768),
                "720x1280 (HD, Portrait)": (1280, 720), "768x1366 (HD, Portrait)": (1366, 768),
                "900x1600 (HD+, Portrait)": (1600, 900), "1080x1920 (Full HD or 1080p, Portrait)": (1920, 1080),
                "1440x2560 (Quad HD or 1440p, Portrait)": (2560, 1440), "2160x3840 (Ultra HD, 4K, or 2160p, Portrait)": (3840, 2160),
                "2880x5120 (5K, Portrait)": (5120, 2880), "4320x7680 (8K, Portrait)": (7680, 4320)
            }
            height, width = standard_sizes.get(standard, (height, width))

        if seed is None:
            seed = random.randint(10**14, 10**15 - 1)
            
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        empty_image = self.generate_empty_image(width, height, batch_size, color)
    
        return (height, width, batch_size, {"samples": latent}, seed, empty_image)


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
    CATEGORY = "GraftingRayman/Images"


    def resize_image(self, image, height, width):
        input_image = image.permute((0, 3, 1, 2))
        resized_image = TF.resize(input_image, (height, width))
        resized_image = resized_image.permute((0, 2, 3, 1))
        return (resized_image,)
        
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
    CATEGORY = "GraftingRayman/Images"


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

class GRResizeImageMethods:
    resize_methods = ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"]
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()  # Assumed to be defined elsewhere
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if not files:
            return {
                "required": {
                    "image": ("IMAGE",),
                    "width": ("INT", {"min": 1}),
                    "height": ("INT", {"min": 1}),
                    "method": (cls.resize_methods,),
                }
            }
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
                "method": (cls.resize_methods,),            
            }
        }

    CATEGORY = "GraftingRayman/Images"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image, width, height, method):
        if image is None:
            raise ValueError("No image file selected.")
        
        image_path = folder_paths.get_annotated_filepath(image)  # Assumed to be defined elsewhere
        img = Image.open(image_path)
        output_images = []
        output_masks = []

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = image.resize((width, height), getattr(Image, method))  # Resize image using the specified method
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), getattr(Image, method))
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, image):
        if image is None:
            return None
        image_path = folder_paths.get_annotated_filepath(image)  # Assumed to be defined elsewhere
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if image is None:
            return "No image file selected."
        if not folder_paths.exists_annotated_filepath(image):  # Assumed to be defined elsewhere
            return "Invalid image file: {}".format(image)
        return True

    def display_image(self, image_tensor):
        image = image_tensor.squeeze().numpy().transpose(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
        plt.show()


class GRImageDetailsSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "GR_"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "image_details"

    OUTPUT_NODE = True

    CATEGORY = "GraftingRayman/Images"


    def image_details(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            img_width, img_height = img.size
            img_mode = img.mode
            img_format = img.format if img.format else "N/A"
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            file_size = img_byte_arr.tell()
            if img_width < 768:
                attributes = [
                    f"Filename: {filename}", f"Type: {self.type}", f"Width: {img_width}",
                    f"Height: {img_height}", f"File size: {file_size} bytes", f"Mode: {img_mode}", f"Format: {img_format}"
                ]
                text_lines = [": ".join(attributes[i:i+3]) for i in range(0, len(attributes), 3)]
            else:
                text_lines = [f"Filename: {filename} : Type: {self.type} : Width: {img_width} : Height: {img_height} : File size: {file_size} bytes : Mode: {img_mode} : Format: {img_format}"]
            base_font_size = 10
            if img_width > 1024:
                extra_pixels = img_width - 1024
                additional_font_size = extra_pixels // 64
                font_size = base_font_size + additional_font_size
            else:
                font_size = base_font_size
            font = ImageFont.truetype("arial.ttf", font_size)
            draw = ImageDraw.Draw(img)
            text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in text_lines) + 10
            new_img = Image.new('RGB', (img_width, img_height + text_height), (0, 0, 0))
            new_img.paste(img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            text_x = 10
            text_y = img_height + 5
            for line in text_lines:
                draw.text((text_x, text_y), line, fill="white", font=font)
                text_y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            new_img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return { "ui": { "images": results } }

class GRImageDetailsDisplayer(GRImageDetailsSave):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    CATEGORY = "GraftingRayman/Images"


    def display_image(self, image_tensor):
        if image_tensor is None:
            raise ValueError("No image tensor provided.")
        
        image = image_tensor.squeeze().numpy().transpose(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
        height, width, _ = image.shape
        image_type = image.dtype
        plt.text(0, -0.1, f"Height: {height}\nWidth: {width}\nType: {image_type}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10, color='white')
        plt.tight_layout()  # Adjust layout to ensure text is visible
        plt.show()

class GRImagePaste:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "opacity": ("INT", {"min": 0, "max": 100, "default": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gr_image_paste"
    CATEGORY = "GraftingRayman/Images"

    def gr_image_paste(self, image1, image2, opacity):
        device = image1.device
        image2 = image2.to(device)
        image2 = TF.resize(image2, image1.shape[-2:])
        opacity_factor = opacity / 100.0
        combined_image = image1 * (1 - opacity_factor) + image2 * opacity_factor
        return (combined_image,)

class GRImagePasteWithMask:
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

    _blend_methods = [
        "add", "subtract", "multiply", "screen", "overlay", "difference", "hard_light", "soft_light",
        "add_modulo", "blend", "darker", "duplicate", "lighter", "subtract_modulo"
    ]

    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", ), "overlay_image": ("IMAGE", ), "mask_image": ("MASK", ),
                "opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "overlay_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "overlay_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "overlay_fit": (["left", "right", "center", "top", "top left", "top right", "bottom", "bottom left", "bottom right"], {"default": "center"}),
                "mask_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_fit": (["left", "right", "center", "top", "top left", "top right", "bottom", "bottom left", "bottom right", "zoom_left", "zoom_center", "zoom_right", "fit"], {"default": "center"}),
                "mask_zoom": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_stretch_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_stretch_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "outline": ("BOOLEAN", {"default": False}),
                "outline_thickness": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "outline_colour": (list(sorted(cls._available_colours.keys())), {"default": "black"}),
                "outline_opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "outline_position": (["center", "inside", "outside"], {"default": "center"}),
                "blend": ("BOOLEAN", {"default": False}),
                "blend_method": (cls._blend_methods, {"default": "add"}),
                "blend_strength": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "blend_area": (["all", "inside", "outside", "outline"], {"default": "all"})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "inverted_mask_image", "contour_image", "image_dimensions")
    FUNCTION = "paste_with_mask"
    CATEGORY = "GraftingRayman\Images"

    def hex_to_rgb(self, hex_color): 
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_rgba(self, hex_color, alpha): 
        r, g, b = self.hex_to_rgb(hex_color)
        return (r, g, b, int(alpha * 255 / 100))

    def blend_images(self, img1, img2, method, strength):
        blended = None
        if method == "add":
            blended = ImageChops.add(img1, img2)
        elif method == "subtract":
            blended = ImageChops.subtract(img1, img2)
        elif method == "multiply":
            blended = ImageChops.multiply(img1, img2)
        elif method == "screen":
            blended = ImageChops.screen(img1, img2)
        elif method == "overlay":
            blended = ImageChops.overlay(img1, img2)
        elif method == "difference":
            blended = ImageChops.difference(img1, img2)
        elif method == "hard_light":
            blended = ImageChops.hard_light(img1, img2)
        elif method == "soft_light":
            blended = ImageChops.soft_light(img1, img2)
        elif method == "add_modulo":
            blended = ImageChops.add_modulo(img1, img2)
        elif method == "blend":
            blended = ImageChops.blend(img1, img2, strength / 100.0)
        elif method == "darker":
            blended = ImageChops.darker(img1, img2)
        elif method == "duplicate":
            blended = ImageChops.duplicate(img1)
        elif method == "lighter":
            blended = ImageChops.lighter(img1, img2)
        elif method == "subtract_modulo":
            blended = ImageChops.subtract_modulo(img1, img2)
        else:
            raise ValueError("Unsupported blend method")
        
        return Image.blend(img1, blended, strength / 100.0)

    def resize_and_fit_image(self, background_image, image, fit_option, x_offset, y_offset, mask_zoom, mask_stretch_x, mask_stretch_y, is_mask=False):
        bg_h, bg_w = background_image.shape[1], background_image.shape[2]
        image_pil = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype(np.uint8))
        if is_mask:
            image_pil = image_pil.convert("L")
        image_w, image_h = image_pil.size

        # Apply mask zoom
        if mask_zoom != 0:
            new_width = image_w + mask_zoom
            new_height = image_h + mask_zoom
            image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            image_w, image_h = new_width, new_height

        # Apply mask stretch x
        if mask_stretch_x != 0:
            new_width = image_w + mask_stretch_x
            image_pil = image_pil.resize((new_width, image_h), Image.LANCZOS)
            image_w = new_width

        # Apply mask stretch y
        if mask_stretch_y != 0:
            new_height = image_h + mask_stretch_y
            image_pil = image_pil.resize((image_w, new_height), Image.LANCZOS)
            image_h = new_height

        if is_mask:
            if fit_option == "fit":
                # Convert image to binary mask
                mask_np = np.array(image_pil) > 0
                # Find bounding box
                coords = np.column_stack(np.where(mask_np))
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox = (x_min, y_min, x_max, y_max)
                # Crop the mask to the bounding box
                image_pil = image_pil.crop(bbox)
                image_w, image_h = image_pil.size

                # Resize to fit the background image while keeping proportions
                scale = min(bg_w / image_w, bg_h / image_h)
                new_size = (int(image_w * scale), int(image_h * scale))
                image_pil = image_pil.resize(new_size, Image.LANCZOS)

                # Center the mask on the background
                left = (bg_w - new_size[0]) // 2 + x_offset
                top = (bg_h - new_size[1]) // 2 + y_offset
                new_image = Image.new("L", (bg_w, bg_h))
                new_image.paste(image_pil, (left, top))
                image_pil = new_image
            elif fit_option in ["zoom_left", "zoom_center", "zoom_right"]:
                if bg_h > bg_w:
                    new_height = bg_h
                    new_width = int((new_height / image_h) * image_w)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    if fit_option == "zoom_left":
                        image_pil = image_pil.crop((0 + x_offset, 0 + y_offset, bg_w + x_offset, bg_h + y_offset))
                    elif fit_option == "zoom_right":
                        image_pil = image_pil.crop((new_width - bg_w + x_offset, 0 + y_offset, new_width + x_offset, bg_h + y_offset))
                    else:  # zoom_center
                        left = (new_width - bg_w) // 2
                        image_pil = image_pil.crop((left + x_offset, 0 + y_offset, left + bg_w + x_offset, bg_h + y_offset))
                else:
                    scale = min(bg_w / image_w, bg_h / image_h)
                    new_size = (int(image_w * scale), int(image_h * scale))
                    image_pil = image_pil.resize(new_size, Image.LANCZOS)
                    left = top = 0
                    if "top" in fit_option:
                        top = 0
                    elif "bottom" in fit_option:
                        top = image_pil.size[1] - bg_h
                    else:
                        top = (image_pil.size[1] - bg_h) // 2

                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = image_pil.size[0] - bg_w
                    else:
                        left = (image_pil.size[0] - bg_w) // 2

                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
            else:
                if bg_h < bg_w and image_h > image_w:
                    new_width = bg_w
                    new_height = int((new_width / image_w) * image_h)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    top = (new_height - bg_h) // 2 if fit_option == "center" else 0 if "top" in fit_option else new_height - bg_h
                    left = 0
                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = new_width - bg_w
                    else:
                        left = (new_width - bg_w) // 2
                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
                elif bg_h > bg_w and image_h < image_w:
                    new_width = bg_w
                    new_height = int((new_width / image_w) * image_h)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    top = (bg_h - new_height) // 2 if fit_option == "center" else 0 if "top" in fit_option else bg_h - new_height
                    left = 0
                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = bg_w - new_width
                    else:
                        left = (bg_w - new_width) // 2
                    new_image = Image.new("L", (bg_w, bg_h))
                    new_image.paste(image_pil, (left + x_offset, top + y_offset))
                    image_pil = new_image
                else:
                    scale = min(bg_w / image_w, bg_h / image_h)
                    new_size = (int(image_w * scale), int(image_h * scale))
                    image_pil = image_pil.resize(new_size, Image.LANCZOS)
                    left = top = 0
                    if "top" in fit_option:
                        top = 0
                    elif "bottom" in fit_option:
                        top = image_pil.size[1] - bg_h
                    else:
                        top = (image_pil.size[1] - bg_h) // 2

                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = image_pil.size[0] - bg_w
                    else:
                        left = (image_pil.size[0] - bg_w) // 2

                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
        else:
            if bg_h < bg_w and image_h > image_w:
                new_width = bg_w
                new_height = int((new_width / image_w) * image_h)
                image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            elif bg_h > bg_w and image_h < image_w:
                new_height = bg_h
                new_width = int((new_height / image_h) * image_w)
                image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            else:
                scale = min(bg_w / image_w, bg_h / image_h)
                new_size = (int(image_w * scale), int(image_h * scale))
                image_pil = image_pil.resize(new_size, Image.LANCZOS)

            left = top = 0
            if "top" in fit_option:
                top = 0
            elif "bottom" in fit_option:
                top = image_pil.size[1] - bg_h
            else:
                top = (image_pil.size[1] - bg_h) // 2

            if "left" in fit_option:
                left = 0
            elif "right" in fit_option:
                left = image_pil.size[0] - bg_w
            else:
                left = (image_pil.size[0] - bg_w) // 2

            image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))

        return torch.tensor(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)

    def paste_with_mask(self, background_image, overlay_image, mask_image, opacity, overlay_x, overlay_y, overlay_fit, mask_x, mask_y, mask_fit, mask_zoom, mask_stretch_x, mask_stretch_y, outline, outline_thickness, outline_colour, outline_opacity, outline_position, blend, blend_method, blend_strength, blend_area):
        overlay_image = self.resize_and_fit_image(background_image, overlay_image, overlay_fit, overlay_x, overlay_y, 0, 0, 0, is_mask=False)
        mask_image = self.resize_and_fit_image(background_image, mask_image, mask_fit, mask_x, mask_y, mask_zoom, mask_stretch_x, mask_stretch_y, is_mask=True)

        mask_image = mask_image.unsqueeze(1)
        mask_image = mask_image.expand(background_image.shape[0], background_image.shape[3], background_image.shape[1], background_image.shape[2]).permute(0, 2, 3, 1)
        opacity /= 100.0
        output_image = (background_image * (1 - mask_image * opacity) + overlay_image * (mask_image * opacity)).clamp(0, 1)
        inverted_mask_image = (background_image * mask_image * opacity + overlay_image * (1 - mask_image * opacity)).clamp(0, 1)

        if blend:
            output_image_pil = Image.fromarray((output_image.squeeze().cpu().numpy() * 255).astype(np.uint8))
            overlay_image_pil = Image.fromarray((overlay_image.squeeze().cpu().numpy() * 255).astype(np.uint8))
            inverted_mask_image_pil = Image.fromarray((inverted_mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8))

            if blend_area == "all":
                output_image_pil = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
            elif blend_area == "inside":
                mask_np = (mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                blended_inside = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_inside, output_image_pil, mask_pil)
                blended_inside_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_inside_inverted, inverted_mask_image_pil, mask_pil)
            elif blend_area == "outside":
                mask_np = ((1 - mask_image).squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                blended_outside = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_outside, output_image_pil, mask_pil)
                blended_outside_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_outside_inverted, inverted_mask_image_pil, mask_pil)
            elif blend_area == "outline":
                mask_np = (mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
                if len(mask_np.shape) > 2: mask_np = mask_np[:, :, 0]
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outline_mask = Image.new("L", (mask_np.shape[1], mask_np.shape[0]), 0)
                outline_draw = ImageDraw.Draw(outline_mask)
                for contour in contours:
                    contour_list = [tuple(point[0]) for point in contour]
                    if len(contour_list) > 2:
                        offset = outline_thickness // 2
                        offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                        outline_draw.line(offset_contour + [offset_contour[0]], fill=255, width=outline_thickness)
                outline_mask = outline_mask.convert("L")
                blended_outline = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_outline, output_image_pil, outline_mask)
                blended_outline_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_outline_inverted, inverted_mask_image_pil, outline_mask)

            output_image = torch.tensor(np.array(output_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            inverted_mask_image = torch.tensor(np.array(inverted_mask_image_pil).astype(np.float32) / 255.0).unsqueeze(0)

        contour_image = torch.zeros_like(output_image)
        if outline:
            mask_np = (mask_image.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            if len(mask_np.shape) > 2: mask_np = mask_np[:, :, 0]
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_colour_rgba = self.hex_to_rgba(self._available_colours[outline_colour], outline_opacity)
            output_image_np = (output_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            output_image_pil, contour_image_pil = Image.fromarray(output_image_np), Image.fromarray(np.zeros_like(output_image_np))
            draw, contour_draw = ImageDraw.Draw(output_image_pil, "RGBA"), ImageDraw.Draw(contour_image_pil, "RGBA")
            for contour in contours:
                contour_list = [tuple(point[0]) for point in contour]
                if len(contour_list) > 2:
                    offset = outline_thickness // 2
                    offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                    draw.line(offset_contour + [offset_contour[0]], fill=outline_colour_rgba, width=outline_thickness)
                    contour_draw.line(contour_list + [contour_list[0]], fill=outline_colour_rgba, width=outline_thickness)
            
            inverted_mask_image_np = (inverted_mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            inverted_mask_image_pil = Image.fromarray(inverted_mask_image_np)
            inverted_draw = ImageDraw.Draw(inverted_mask_image_pil, "RGBA")
            for contour in contours:
                contour_list = [tuple(point[0]) for point in contour]
                if len(contour_list) > 2:
                    offset = outline_thickness // 2
                    offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                    inverted_draw.line(offset_contour + [offset_contour[0]], fill=outline_colour_rgba, width=outline_thickness)

            output_image = torch.tensor(np.array(output_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            contour_image = torch.tensor(np.array(contour_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            inverted_mask_image = torch.tensor(np.array(inverted_mask_image_pil).astype(np.float32) / 255.0).unsqueeze(0)

        image_dimensions = f"Width: {background_image.shape[2]}, Height: {background_image.shape[1]}"
        
        return output_image, inverted_mask_image, contour_image, image_dimensions


class GRBackgroundRemoverREMBG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "rembg_model": (
                    ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"],
                    {"default": "u2net"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    CATEGORY = "GraftingRayman\Images"
    FUNCTION = "remove_background"

    def remove_background(self, image, rembg_model="u2net"):
        return self._rembg_remove_background_wrapper(image, rembg_model)

    def _rembg_remove_background_wrapper(self, image, rembg_model):
        session = new_session(model_name=rembg_model)
        return self._rembg_remove_background(image, session)

    def _rembg_remove_background(self, image, session):
        image = image.permute([0, 3, 1, 2])  # (B, H, W, C) to (B, C, H, W)
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg_remove(img, session=session)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])  # (B, C, H, W) to (B, H, W, C)
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return output[:, :, :, :3], mask
