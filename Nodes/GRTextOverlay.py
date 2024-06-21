import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import math
import random
from comfy.utils import ProgressBar

class GRTextOverlay:
    _horizontal_alignments = ["left", "center", "right"]
    _vertical_alignments = ["top", "middle", "bottom"]
    _justifications = ["left", "center", "right"]

    @staticmethod
    def _populate_fonts_from_os():
        fonts = set()
        font_paths = [
            "/usr/share/fonts/truetype",      # Common Linux path
            "/usr/local/share/fonts",         # Another common Linux path
            "/Library/Fonts",                 # macOS path
            "/System/Library/Fonts",          # macOS system path
            "C:\\Windows\\Fonts"              # Windows path
        ]
        for path in font_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".ttf") or file.endswith(".otf"):
                            fonts.add(os.path.join(root, file))
        return sorted(list(fonts))

    _available_fonts = _populate_fonts_from_os()
    
    _available_colours = {
        "amethyst": "#9966CC",
        "black": "#000000",
        "blue": "#0000FF",
        "cyan": "#00FFFF",
        "diamond": "#B9F2FF",
        "emerald": "#50C878",
        "gold": "#FFD700",
        "gray": "#808080",
        "green": "#008000",
        "lime": "#00FF00",
        "magenta": "#FF00FF",
        "maroon": "#800000",
        "navy": "#000080",
        "neon_blue": "#1B03A3",
        "neon_green": "#39FF14",
        "neon_orange": "#FF6103",
        "neon_pink": "#FF10F0",
        "neon_yellow": "#DFFF00",
        "olive": "#808000",
        "platinum": "#E5E4E2",
        "purple": "#800080",
        "red": "#FF0000",
        "rose_gold": "#B76E79",
        "ruby": "#E0115F",
        "sapphire": "#0F52BA",
        "silver": "#C0C0C0",
        "teal": "#008080",
        "topaz": "#FFCC00",
        "white": "#FFFFFF",
        "yellow": "#FFFF00"
    }

    def __init__(self, device="cpu"):
        self.device = device
        self._loaded_font = None
        self._full_text = None
        self._x = None
        self._y = None
        self.fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Hello"},
                ),
                "font": (
                    cls._available_fonts,
                    {"default": cls._available_fonts[0] if cls._available_fonts else "arial.ttf"},
                ),
                "font_size": (
                    "INT",
                    {"default": 32, "min": 1, "max": 256, "step": 1},
                ),
                "vertical_alignment": (
                    cls._vertical_alignments,
                    {"default": "bottom"},
                ),
                "horizontal_alignment": (
                    cls._horizontal_alignments,
                    {"default": "center"},
                ),
                "justification": (
                    cls._justifications,
                    {"default": "left"},
                ),
                "padding": (
                    "INT",
                    {"default": 16, "min": 0, "max": 128, "step": 1},
                ),
                "stroke_thickness": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "stroke_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "fill_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "white"},
                ),
                "line_spacing": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 200, "step": 0.01},
                ),
                "letter_spacing": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 200, "step": 0.01},
                ),
                "x_align": (
                    "INT",
                    {"default": 0, "min": -128, "max": 128, "step": 1},
                ),
                "y_align": (
                    "INT",
                    {"default": 0, "min": -128, "max": 128, "step": 1},
                ),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    FUNCTION = "batch_process"
    CATEGORY = "GraftingRayman/Overlays"


    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = hex_color * 2
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def load_font(self, font, font_size):
        """Load the specified font."""
        try:
            return ImageFont.truetype(font, font_size)
        except Exception as e:
            print(f"Error loading font: {e}... Using default font")
            return ImageFont.load_default()

    def draw_text_with_letter_spacing(
        self, draw, position, text, font, fill, stroke_fill, stroke_width, align, letter_spacing, spacing, justification
    ):
        """Draw text with letter spacing."""
        x, y = position
        lines = text.split('\n')
        for line in lines:
            line_width = draw.textlength(line, font=font)
            if justification == "center":
                x = position[0] - line_width / 2
            elif justification == "right":
                x = position[0] - line_width
            else:
                x = position[0]

            for char in line:
                draw.text((x, y), char, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width, align=align)
                bbox = font.getbbox(char)
                x += (bbox[2] - bbox[0]) + letter_spacing

            y += font.size + spacing

    def create_masks(self, image, text, font, font_size, stroke_thickness, justification, line_spacing, padding):
        """Create two masks: one for text only and one including stroke."""
        # Create a mask for text only
        text_mask = Image.new("L", (image.width, image.height), 0)
        text_mask_draw = ImageDraw.Draw(text_mask)
        self.draw_text_with_letter_spacing(
            text_mask_draw,
            (self._x, self._y),
            text,
            font,
            255,
            None,
            0,
            justification,
            0,
            line_spacing,
            justification
        )
        
        # Create a mask for text with stroke
        stroke_mask = Image.new("L", (image.width, image.height), 0)
        stroke_mask_draw = ImageDraw.Draw(stroke_mask)
        self.draw_text_with_letter_spacing(
            stroke_mask_draw,
            (self._x, self._y),
            text,
            font,
            255,
            255,
            int(font_size * stroke_thickness * 0.5),
            justification,
            0,
            line_spacing,
            justification
        )
        
        text_mask_tensor = torch.tensor(np.array(text_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        stroke_mask_tensor = torch.tensor(np.array(stroke_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        return text_mask_tensor, stroke_mask_tensor

    def draw_text(
        self,
        image,
        text,
        font_size,
        font,
        fill_colour,
        stroke_colour,
        stroke_thickness,
        padding,
        horizontal_alignment,
        vertical_alignment,
        justification,
        x_align,
        y_align,
        line_spacing,
        letter_spacing,
        use_cache=False,
    ):
        """Draw the text on the image with the specified parameters."""
        self._loaded_font = self.load_font(font, font_size)
        draw = ImageDraw.Draw(image)
        self._full_text = text

        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            self._full_text,
            font=self._loaded_font,
            stroke_width=int(font_size * stroke_thickness * 0.5),
            spacing=int(font_size * line_spacing),
            align=horizontal_alignment,
        )

        if horizontal_alignment == "left":
            self._x = padding
        elif horizontal_alignment == "center":
            self._x = image.width / 2
        elif horizontal_alignment == "right":
            self._x = image.width - padding

        self._x += x_align
        if vertical_alignment == "middle":
            self._y = (image.height - (bottom - top)) / 2
        elif vertical_alignment == "top":
            self._y = padding
        elif vertical_alignment == "bottom":
            self._y = image.height - (bottom - top) - padding
        self._y += y_align

        self.draw_text_with_letter_spacing(
            draw,
            (self._x, self._y),
            self._full_text,
            self._loaded_font,
            self.hex_to_rgb(self._available_colours[fill_colour]),
            self.hex_to_rgb(self._available_colours[stroke_colour]),
            int(font_size * stroke_thickness * 0.5),
            horizontal_alignment,
            letter_spacing,
            line_spacing,
            justification
        )

        text_mask, stroke_mask = self.create_masks(image, self._full_text, self._loaded_font, font_size, stroke_thickness, justification, line_spacing, padding)

        return image, text_mask, stroke_mask

    def batch_process(
        self,
        image,
        text,
        font_size,
        font,
        fill_colour,
        stroke_colour,
        stroke_thickness,
        padding,
        horizontal_alignment,
        vertical_alignment,
        justification,
        x_align,
        y_align,
        line_spacing,
        letter_spacing
    ):
        """Batch process images."""
        if len(image.shape) == 3:
            image_np = image.cpu().numpy()
            image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
            image, text_mask, stroke_mask = self.draw_text(
                image,
                text,
                font_size,
                font,
                fill_colour,
                stroke_colour,
                stroke_thickness,
                padding,
                horizontal_alignment,
                vertical_alignment,
                justification,
                x_align,
                y_align,
                line_spacing,
                letter_spacing
            )
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            return image_tensor_out, text_mask, stroke_mask
        else:
            image_np = image.cpu().numpy()
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_np]
            images_out, text_masks_out, stroke_masks_out = [], [], []
            for img in images:
                img, text_mask, stroke_mask = self.draw_text(
                    img,
                    text,
                    font_size,
                    font,
                    fill_colour,
                    stroke_colour,
                    stroke_thickness,
                    padding,
                    horizontal_alignment,
                    vertical_alignment,
                    justification,
                    x_align,
                    y_align,
                    line_spacing,
                    letter_spacing,
                    use_cache=False
                )
                images_out.append(np.array(img).astype(np.float32) / 255.0)
                text_masks_out.append(text_mask)
                stroke_masks_out.append(stroke_mask)

            images_tensor = torch.from_numpy(np.stack(images_out))
            text_masks_tensor = torch.from_numpy(np.stack([mask.numpy() for mask in text_masks_out]))
            stroke_masks_tensor = torch.from_numpy(np.stack([mask.numpy() for mask in stroke_masks_out]))

            return images_tensor, text_masks_tensor, stroke_masks_tensor



class GROnomatopoeia:
    _horizontal_alignments = ["left", "center", "right"]
    _vertical_alignments = ["top", "middle", "bottom"]
    _justifications = ["left", "center", "right"]
    _onomatopoeias = ["BANG", "BOOM", "CRASH", "SPLASH", "WHIZZ", "POP", "CLANG", "BUZZ", "ZAP", "WHOOSH", 
                      "BAM", "BAMBAM", "POW", "EEEEEEE", "SPLAT", "SQUISH", "WOW", "POW", "WTF?", "OMG", 
                      "SMASH", "HAHA", "OOOPS", "WHAM", "POOF", "YEAH", "LOL", "BOINK", "ZOIKS", "PEW PEW", 
                      "BRRRRP"]

    @staticmethod
    def _populate_fonts_from_os():
        fonts = set()
        font_paths = ["/usr/share/fonts/truetype", "/usr/local/share/fonts", "/Library/Fonts", "/System/Library/Fonts", "C:\\Windows\\Fonts"]
        for path in font_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".ttf") or file.endswith(".otf"):
                            fonts.add(os.path.join(root, file))
        return sorted(list(fonts))

    _available_fonts = _populate_fonts_from_os()
    _available_colours = {
        "amethyst": "#9966CC", "black": "#000000", "blue": "#0000FF", "cyan": "#00FFFF", "diamond": "#B9F2FF",
        "emerald": "#50C878", "gold": "#FFD700", "gray": "#808080", "green": "#008000", "lime": "#00FF00",
        "magenta": "#FF00FF", "maroon": "#800000", "navy": "#000080", "neon_blue": "#1B03A3", "neon_green": "#39FF14",
        "neon_orange": "#FF6103", "neon_pink": "#FF10F0", "neon_yellow": "#DFFF00", "olive": "#808000", "platinum": "#E5E4E2",
        "purple": "#800080", "red": "#FF0000", "rose_gold": "#B76E79", "ruby": "#E0115F", "sapphire": "#0F52BA",
        "silver": "#C0C0C0", "teal": "#008080", "topaz": "#FFCC00", "white": "#FFFFFF", "yellow": "#FFFF00"
    }

    def __init__(self, device="cpu"):
        self.device = device
        self._loaded_font = None
        self._full_text = None
        self._x = None
        self._y = None
        self.fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello"}),
                "randomize": ("BOOLEAN", {"default": False}),
                "font": (cls._available_fonts, {"default": cls._available_fonts[0] if cls._available_fonts else "arial.ttf"}),
                "font_size": ("INT", {"default": 100, "min": 1, "max": 256, "step": 1}),
                "vertical_alignment": (cls._vertical_alignments, {"default": "middle"}),
                "horizontal_alignment": (cls._horizontal_alignments, {"default": "center"}),
                "justification": (cls._justifications, {"default": "left"}),
                "bubble_justification": (cls._justifications, {"default": "center"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "stroke_thickness": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "stroke_colour": (list(sorted(cls._available_colours.keys())), {"default": "red"}),
                "fill_colour": (list(sorted(cls._available_colours.keys())), {"default": "purple"}),
                "bubble": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "bubble_distance": ("INT", {"default": 100, "min": 0, "max": 500, "step": 1}),
                "bubble_colour": (list(sorted(cls._available_colours.keys())), {"default": "maroon"}),
                "bubble_stroke_thickness": ("INT", {"default": 10, "min": 0, "max": 20, "step": 1}),
                "bubble_fill": ("BOOLEAN", {"default": True}),
                "bubble_fill_colour": (list(sorted(cls._available_colours.keys())), {"default": "white"}),
                "jagged_points": ("INT", {"default": 45, "min": 3, "max": 100, "step": 1}),
                "jagged_min_distance": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "jagged_max_distance": ("INT", {"default": 100, "min": 0, "max": 500, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999999999}),
                "vertical_randomness": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "letter_spacing": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "randomize_colours": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_process"
    CATEGORY = "GraftingRayman/Overlays"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = hex_color * 2
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

    def load_font(self, font, font_size):
        try:
            return ImageFont.truetype(font, font_size)
        except Exception as e:
            print(f"Error loading font: {e}... Using default font")
            return ImageFont.load_default()

    def draw_text_with_letter_spacing(self, draw, position, text, font, fill, stroke_fill, stroke_width, align, vertical_randomness, letter_spacing, justification):
        x, y = position
        lines = text.split('\n')
        for line in lines:
            line_width = draw.textlength(line, font=font)
            if justification == "center":
                x = position[0] - line_width / 2
            elif justification == "right":
                x = position[0] - line_width
            else:
                x = position[0]
            for char in line:
                y_random = y + random.randint(-vertical_randomness, vertical_randomness)
                draw.text((x, y_random), char, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width, align=align)
                bbox = font.getbbox(char)
                x += (bbox[2] - bbox[0]) + letter_spacing
            y += font.size

    def draw_bubble(self, draw, bubble_bbox, bubble_thickness, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, horizontal_alignment, vertical_alignment, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, font_size, image_width, image_height):
        left, top, right, bottom = bubble_bbox
        width, height = right - left, bottom - top
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        half_font_size = font_size / 2

        if horizontal_alignment == "left" or bubble_justification == "left":
            center_x = bubble_distance + jagged_max_distance + bubble_stroke_thickness + half_font_size
        elif horizontal_alignment == "center":
            center_x = image_width / 2
        elif horizontal_alignment == "right" or bubble_justification == "right":
            center_x = image_width - bubble_distance - jagged_max_distance - bubble_stroke_thickness - half_font_size

        if vertical_alignment == "top":
            center_y = bubble_distance + jagged_max_distance + bubble_stroke_thickness + half_font_size
        elif vertical_alignment == "middle":
            center_y = image_height / 2
        elif vertical_alignment == "bottom":
            center_y = image_height - bubble_distance - jagged_max_distance - bubble_stroke_thickness - half_font_size

        bubble_points = []
        angle_step = 2 * math.pi / jagged_points

        for i in range(jagged_points):
            angle = i * angle_step
            random_distance = random.randint(jagged_min_distance, jagged_max_distance)
            radius_x = width / 2 + bubble_distance + (random_distance if i % 2 == 0 else 0)
            radius_y = height / 2 + bubble_distance + (random_distance if i % 2 == 0 else 0)
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            bubble_points.append((x, y))

        if bubble_fill:
            draw.polygon(bubble_points, fill=self.hex_to_rgb(bubble_fill_colour), outline=self.hex_to_rgb(bubble_colour), width=bubble_stroke_thickness)
        else:
            draw.polygon(bubble_points, outline=self.hex_to_rgb(bubble_colour), width=bubble_stroke_thickness)

    def randomize_colours(self):
        return random.choice(list(self._available_colours.values()))

    def draw_text(self, image, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours):
        self._loaded_font = self.load_font(font, font_size)
        draw = ImageDraw.Draw(image)
        
        if not text:
            random_exclamations = "!" * random.randint(1, 3)
            self._full_text = random.choice(self._onomatopoeias) + random_exclamations
        else:
            if randomize:
                text = ''.join(random.sample(text.upper(), len(text)))
            random_exclamations = "!" * random.randint(1, 3)
            self._full_text = text + random_exclamations
        
        random.seed(seed)
        
        if randomize_colours:
            fill_colour = self.randomize_colours()
            stroke_colour = self.randomize_colours()
            bubble_colour = self.randomize_colours()
            bubble_fill_colour = self.randomize_colours()
        else:
            fill_colour = self._available_colours[fill_colour]
            stroke_colour = self._available_colours[stroke_colour]
            bubble_colour = self._available_colours[bubble_colour]
            bubble_fill_colour = self._available_colours[bubble_fill_colour]

        image_width, image_height = image.size

        left, top, right, bottom = draw.textbbox((0, 0), self._full_text, font=self._loaded_font, stroke_width=stroke_thickness, align=horizontal_alignment)

        if bubble > 0:
            self.draw_bubble(draw, (left, top, right, bottom), bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, horizontal_alignment, vertical_alignment, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, font_size, image_width, image_height)

        text_left, text_top, text_right, text_bottom = left, top, right, bottom

        if horizontal_alignment == "left":
            self._x = padding + bubble_distance + jagged_max_distance + stroke_thickness
        elif horizontal_alignment == "center":
            self._x = (image_width - (text_right - text_left)) / 2
        elif horizontal_alignment == "right":
            self._x = image_width - (text_right - text_left) - padding - bubble_distance - jagged_max_distance - stroke_thickness

        if vertical_alignment == "middle":
            self._y = (image_height - (text_bottom - text_top)) / 2
        elif vertical_alignment == "top":
            self._y = padding + bubble_distance + jagged_max_distance + stroke_thickness
        elif vertical_alignment == "bottom":
            self._y = image_height - (text_bottom - text_top) - padding - bubble_distance - jagged_max_distance - stroke_thickness

        self.draw_text_with_letter_spacing(draw, (self._x, self._y), self._full_text, self._loaded_font, self.hex_to_rgb(fill_colour), self.hex_to_rgb(stroke_colour), stroke_thickness, horizontal_alignment, vertical_randomness, letter_spacing, justification)
        return image

    def batch_process(self, image, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours):
        pbar = ProgressBar(len(image))
        if len(image.shape) == 3:
            image_np = image.cpu().numpy()
            image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
            if seed == 0:
                seed = random.randint(10**14, 10**15 - 1)
            random.seed(seed)
            if randomize_colours:
                fill_colour = self.randomize_colours()
                stroke_colour = self.randomize_colours()
                bubble_colour = self.randomize_colours()
                bubble_fill_colour = self.randomize_colours()
            image = self.draw_text(image, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours)
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            pbar.update_absolute(0)
            return image_tensor_out,
        else:
            image_np = image.cpu().numpy()
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_np]
            images_out = []
            for idx, img in enumerate(images):
                if seed == 0:
                    seed = random.randint(10**14, 10**15 - 1)
                random.seed(seed + idx)  # Different seed for each image in the batch
                if randomize_colours:
                    fill_colour = self.randomize_colours()
                    stroke_colour = self.randomize_colours()
                    bubble_colour = self.randomize_colours()
                    bubble_fill_colour = self.randomize_colours()
                img = self.draw_text(img, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, bubble_justification, jagged_points, jagged_min_distance, jagged_max_distance, seed + idx, vertical_randomness, letter_spacing, randomize_colours)
                images_out.append(np.array(img).astype(np.float32) / 255.0)
                pbar.update_absolute(idx)
            images_tensor = torch.from_numpy(np.stack(images_out))
            return images_tensor,
