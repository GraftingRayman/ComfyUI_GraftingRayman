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
                "(SD) 512x512": (512, 512), "(SD2) 768x768": (768, 768), "(SD2) 768x512": (768, 512),
                "(SD2) 512x768 (Portrait)": (512, 768), "(SDXL) 1024x1024": (1024, 1024), "640x480 (VGA)": (640, 480),
                "800x600 (SVGA)": (800, 608), "960x544 (Half HD)": (960, 544), "1024x768 (XGA)": (1024, 768),
                "1280x720 (HD)": (1280, 720), "1366x768 (HD)": (1360, 768), "1600x900 (HD+)": (1600, 896),
                "1920x1080 (Full HD or 1080p)": (1920, 1088), "2560x1440 (Quad HD or 1440p)": (2560, 1440),
                "3840x2160 (Ultra HD, 4K, or 2160p)": (3840, 2160), "5120x2880 (5K)": (5120, 2880),
                "7680x4320 (8K)": (7680, 4320), "480x640 (VGA, Portrait)": (480, 640), "600x800 (SVGA, Portrait)": (600, 800),
                "544x960 (Half HD, Portrait)": (544, 960), "768x1024 (XGA, Portrait)": (768, 1024),
                "720x1280 (HD, Portrait)": (720, 1280), "768x1366 (HD, Portrait)": (768, 1366),
                "900x1600 (HD+, Portrait)": (900, 1600), "1080x1920 (Full HD or 1080p, Portrait)": (1080, 1920),
                "1440x2560 (Quad HD or 1440p, Portrait)": (1440, 2560), "2160x3840 (Ultra HD, 4K, or 2160p, Portrait)": (2160, 3840),
                "2880x5120 (5K, Portrait)": (2880, 5120), "4320x7680 (8K, Portrait)": (4320, 7680)
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
