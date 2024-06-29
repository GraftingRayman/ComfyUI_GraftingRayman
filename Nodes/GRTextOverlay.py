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
    _edge_styles = ["straight", "curved", "rounded", "wavy", "zigzag", "double_line", "dashed_line", "beveled"]

    @staticmethod
    def _populate_fonts_from_os():
        fonts = set()
        font_paths = [
            "/usr/share/fonts/truetype",
            "/usr/local/share/fonts",
            "/Library/Fonts",
            "/System/Library/Fonts",
            "C:\\Windows\\Fonts"
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
                    "INT",
                    {"default": 2, "min": 0, "max": 50, "step": 1},
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
                "background": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "background_size": (
                    "INT",
                    {"default": 10, "min": 0, "max": 256, "step": 1},
                ),
                "background_stroke": (
                    "INT",
                    {"default": 2, "min": 0, "max": 50, "step": 1},
                ),
                "background_stroke_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "background_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "white"},
                ),
                "background_opacity": (
                    "INT",
                    {"default": 100, "min": 0, "max": 100, "step": 1},
                ),
                "edge_style": (
                    cls._edge_styles,
                    {"default": "straight"},
                ),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    FUNCTION = "batch_process"
    CATEGORY = "GraftingRayman/Overlays"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = hex_color * 2
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def hex_to_rgba(self, hex_color, alpha):
        r, g, b = self.hex_to_rgb(hex_color)
        return (r, g, b, int((100 - alpha) * 255 / 100))

    def load_font(self, font, font_size):
        try:
            return ImageFont.truetype(font, font_size)
        except Exception as e:
            print(f"Error loading font: {e}... Using default font")
            return ImageFont.load_default()

    def draw_text_with_letter_spacing(
        self, draw, position, text, font, fill, stroke_fill, stroke_width, align, letter_spacing, spacing, justification
    ):
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

    def calculate_text_bbox(self, text, font, line_spacing, letter_spacing, justification):
        lines = text.split('\n')
        max_width = 0
        total_height = 0
        line_widths = []

        for line in lines:
            line_width = 0
            for char in line:
                bbox = font.getbbox(char)
                line_width += (bbox[2] - bbox[0]) + letter_spacing
            line_width -= letter_spacing
            max_width = max(max_width, line_width)
            total_height += font.size + line_spacing
            line_widths.append(line_width)

        total_height -= line_spacing
        return max_width, total_height, line_widths

    def is_italic_font(self, font):
        font_name = font.getname()[0].lower()
        font_path = font.path.lower()
        return "italic" in font_name or "oblique" in font_name or "italic" in font_path or "oblique" in font_path

    def adjust_for_italic(self, font):
        if self.is_italic_font(font):
            return font.size // 4
        return 0

    def draw_wavy_edge(self, draw, left, top, right, bottom, fill, outline, width):
        path = []
        wave_height = (bottom - top) / 10
        for x in range(left, right, int((right - left) / 20)):
            path.append((x, top if (x // 20) % 2 == 0 else top + wave_height))
        for x in range(right, left, -int((right - left) / 20)):
            path.append((x, bottom if (x // 20) % 2 == 0 else bottom - wave_height))
        draw.polygon(path, fill=fill, outline=outline, width=width)

    def draw_zigzag_edge(self, draw, left, top, right, bottom, fill, outline, width):
        path = []
        zigzag_height = (bottom - top) / 10
        for x in range(left, right, int((right - left) / 20)):
            path.append((x, top if (x // 20) % 2 == 0 else top + zigzag_height))
        for x in range(right, left, -int((right - left) / 20)):
            path.append((x, bottom if (x // 20) % 2 == 0 else bottom - zigzag_height))
        draw.polygon(path, fill=fill, outline=outline, width=width)

    def draw_background(self, draw, left, top, right, bottom, fill, outline, width, style, italic_adjust):
        left -= italic_adjust
        right += italic_adjust

        if style == "straight":
            draw.rectangle([left, top, right, bottom], fill=fill, outline=outline, width=width)
        elif style == "curved":
            draw.rounded_rectangle([left, top, right, bottom], radius=(bottom - top) / 2, fill=fill, outline=outline, width=width)
        elif style == "rounded":
            draw.rounded_rectangle([left, top, right, bottom], radius=width, fill=fill, outline=outline, width=width)
        elif style == "wavy":
            self.draw_wavy_edge(draw, left, top, right, bottom, fill, outline, width)
        elif style == "zigzag":
            self.draw_zigzag_edge(draw, left, top, right, bottom, fill, outline, width)
        elif style == "double_line":
            draw.rectangle([left, top, right, bottom], fill=fill, outline=outline, width=width)
            draw.rectangle([left + width*2, top + width*2, right - width*2, bottom - width*2], outline=outline, width=width)
        elif style == "dashed_line":
            for i in range(left, right, 10):
                draw.line((i, top, i + 5, top), fill=outline, width=width)
                draw.line((i, bottom, i + 5, bottom), fill=outline, width=width)
            for i in range(top, bottom, 10):
                draw.line((left, i, left, i + 5), fill=outline, width=width)
                draw.line((right, i, right, i + 5), fill=outline, width=width)
        elif style == "beveled":
            draw.polygon([left, top, left + width, top + width, right - width, top + width, right, top,
                          right, bottom, right - width, bottom - width, left + width, bottom - width, left, bottom], 
                         fill=fill, outline=outline, width=width)

    def create_masks(self, image, text, font, font_size, stroke_thickness, justification, line_spacing, padding):
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
        
        stroke_mask = Image.new("L", (image.width, image.height), 0)
        stroke_mask_draw = ImageDraw.Draw(stroke_mask)
        self.draw_text_with_letter_spacing(
            stroke_mask_draw,
            (self._x, self._y),
            text,
            font,
            255,
            255,
            stroke_thickness,
            justification,
            0,
            line_spacing,
            justification
        )
        
        text_mask_tensor = torch.tensor(np.array(text_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        stroke_mask_tensor = torch.tensor(np.array(stroke_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        return text_mask_tensor, stroke_mask_tensor

    def get_bounding_box(self, mask_tensor):
        mask_np = mask_tensor.numpy().squeeze()
        coords = np.column_stack(np.where(mask_np > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return x_min, y_min, x_max, y_max

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
        background,
        background_size,
        background_stroke,
        background_stroke_colour,
        background_colour,
        background_opacity,
        edge_style,
        use_cache=False,
    ):
        self._loaded_font = self.load_font(font, font_size)

        bg_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_layer)

        draw = ImageDraw.Draw(image)
        self._full_text = text

        text_width, text_height, line_widths = self.calculate_text_bbox(
            self._full_text, self._loaded_font, line_spacing, letter_spacing, justification
        )

        italic_adjust = self.adjust_for_italic(self._loaded_font)

        total_padding = padding + stroke_thickness + background_size

        if horizontal_alignment == "left":
            self._x = total_padding
        elif horizontal_alignment == "center":
            self._x = image.width / 2
        elif horizontal_alignment == "right":
            self._x = image.width - total_padding

        self._x += x_align
        if vertical_alignment == "middle":
            self._y = (image.height - text_height) / 2
        elif vertical_alignment == "top":
            self._y = total_padding
        elif vertical_alignment == "bottom":
            self._y = image.height - text_height - total_padding
        self._y += y_align

        _, stroke_mask_tensor = self.create_masks(image, self._full_text, self._loaded_font, font_size, stroke_thickness, justification, line_spacing, padding)
        x_min, y_min, x_max, y_max = self.get_bounding_box(stroke_mask_tensor)

        if background:
            bg_left = x_min - background_size
            bg_top = y_min - background_size
            bg_right = x_max + background_size
            bg_bottom = y_max + background_size

            self.draw_background(
                bg_draw,
                bg_left,
                bg_top,
                bg_right,
                bg_bottom,
                fill=self.hex_to_rgba(self._available_colours[background_colour], background_opacity),
                outline=self.hex_to_rgb(self._available_colours[background_stroke_colour]),
                width=background_stroke,
                style=edge_style,
                italic_adjust=italic_adjust
            )

        image = Image.alpha_composite(image.convert("RGBA"), bg_layer)

        draw = ImageDraw.Draw(image)

        self.draw_text_with_letter_spacing(
            draw,
            (self._x, self._y),
            self._full_text,
            self._loaded_font,
            self.hex_to_rgb(self._available_colours[fill_colour]),
            self.hex_to_rgb(self._available_colours[stroke_colour]),
            stroke_thickness,
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
        letter_spacing,
        background,
        background_size,
        background_stroke,
        background_stroke_colour,
        background_colour,
        background_opacity,
        edge_style
    ):
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
                letter_spacing,
                background,
                background_size,
                background_stroke,
                background_stroke_colour,
                background_colour,
                background_opacity,
                edge_style
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
                    background,
                    background_size,
                    background_stroke,
                    background_stroke_colour,
                    background_colour,
                    background_opacity,
                    edge_style,
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

    _shapes = ["none", "star", "cloud", "lightning", "bomb", "skull", "dots"]

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
                "line_type": (["straight", "curved inwards", "curved outwards", "splat", "drip"], {"default": "straight"}),
                "jagged_points": ("INT", {"default": 45, "min": 3, "max": 100, "step": 2}),
                "jagged_min_distance": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "jagged_max_distance": ("INT", {"default": 100, "min": 0, "max": 500, "step": 1}),
                "shape_type": (cls._shapes, {"default": "none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999999999}),
                "vertical_randomness": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "letter_spacing": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "randomize_colours": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "batch_process"
    CATEGORY = "GraftingRayman/Overlays"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = hex_color * 2
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

    def get_rgb(self, color_name):
        if color_name in self._available_colours:
            return self.hex_to_rgb(self._available_colours[color_name])
        return self.hex_to_rgb(color_name)

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

    def draw_star(self, draw, position, size, color):
        x, y = position
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            angle = (i + 0.5) * 2 * math.pi / 5 - math.pi / 2
            points.append((x + size / 2 * math.cos(angle), y + size / 2 * math.sin(angle)))
        draw.polygon(points, fill=color, outline=color)

    def draw_cloud(self, draw, position, size, color):
        x, y = position
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=color)
        draw.ellipse([x - size / 1.5, y - size * 1.5, x + size / 1.5, y + size / 1.5], fill=color, outline=color)
        draw.ellipse([x + size / 1.5, y - size, x + size * 1.5, y + size], fill=color, outline=color)

    def draw_lightning(self, draw, position, size, color):
        x, y = position
        lightning_patterns = [
            [(x, y), (x + size / 3, y + size / 3), (x - size / 3, y + size / 2), (x + size / 3, y + size), (x - size / 3, y + size)],
            [(x, y), (x + size / 2, y + size / 4), (x - size / 4, y + size / 2), (x + size / 2, y + size), (x - size / 4, y + size)],
            [(x, y), (x + size / 4, y + size / 3), (x - size / 4, y + size / 2), (x + size / 4, y + size), (x - size / 4, y + size)]
        ]
        pattern = random.choice(lightning_patterns)
        draw.polygon(pattern, fill=color, outline=color)

    def draw_bomb(self, draw, position, size, color):
        x, y = position
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=color)
        draw.rectangle([x - size / 5, y - size - size / 2, x + size / 5, y - size], fill=color, outline=color)

    def draw_skull(self, draw, position, size, color):
        x, y = position
        # Skull base
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=color)
        # Eyes
        eye_size = size // 4
        draw.ellipse([x - size / 3 - eye_size, y - size / 4 - eye_size, x - size / 3 + eye_size, y - size / 4 + eye_size], fill="black")
        draw.ellipse([x + size / 3 - eye_size, y - size / 4 - eye_size, x + size / 3 + eye_size, y - size / 4 + eye_size], fill="black")
        # Nose
        draw.polygon([(x, y + size / 10), (x - size / 10, y + size / 5), (x + size / 10, y + size / 5)], fill="black")
        # Mouth
        draw.rectangle([x - size / 6, y + size / 2.5, x + size / 6, y + size / 2], fill="black")

    def draw_dots(self, draw, position, size, color):
        x, y = position
        for i in range(5):
            draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=color)
            x += size * 2

    def draw_shape(self, draw, shape, position, size):
        color = self.randomize_colours()
        if shape == "star":
            self.draw_star(draw, position, size, color)
        elif shape == "cloud":
            self.draw_cloud(draw, position, size, color)
        elif shape == "lightning":
            self.draw_lightning(draw, position, size, color)
        elif shape == "bomb":
            self.draw_bomb(draw, position, size, color)
        elif shape == "skull":
            self.draw_skull(draw, position, size, color)
        elif shape == "dots":
            self.draw_dots(draw, position, size, color)

    def draw_bubble(self, draw, bubble_bbox, bubble_thickness, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance):
        left, top, right, bottom = bubble_bbox
        width, height = right - left, bottom - top
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        bubble_points = []
        control_points_inwards = []
        control_points_outwards = []
        splat_control_points = []
        drip_points = []
        angle_step = 2 * math.pi / jagged_points

        for i in range(jagged_points):
            angle = i * angle_step
            random_distance = random.randint(jagged_min_distance, jagged_max_distance)
            radius_x = width / 2 + bubble_distance + (random_distance if i % 2 == 0 else 0)
            radius_y = height / 2 + bubble_distance + (random_distance if i % 2 == 0 else 0)
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            bubble_points.append((x, y))

            if line_type != "straight":
                next_angle = (i + 1) * angle_step
                mid_x = center_x + radius_x * math.cos((angle + next_angle) / 2)
                mid_y = center_y + radius_y * math.sin((angle + next_angle) / 2)
                control_x_inwards = center_x + (radius_x - random_distance / 2) * math.cos((angle + next_angle) / 2)
                control_y_inwards = center_y + (radius_y - random_distance / 2) * math.sin((angle + next_angle) / 2)
                control_x_outwards = center_x + (radius_x + random_distance / 2) * math.cos((angle + next_angle) / 2)
                control_y_outwards = center_y + (radius_y + random_distance / 2) * math.sin((angle + next_angle) / 2)

                control_points_inwards.append((control_x_inwards, control_y_inwards))
                control_points_outwards.append((control_x_outwards, control_y_outwards))

                # Control points for splat effect
                splat_control_x = center_x + (radius_x + random_distance) * math.cos((angle + next_angle) / 2)
                splat_control_y = center_y + (radius_y + random_distance) * math.sin((angle + next_angle) / 2)
                splat_control_points.append((splat_control_x, splat_control_y))

                # Points for drip effect
                drip_length = random.randint(10, 30)
                drip_x = x + drip_length * math.cos(angle + math.pi / 2)
                drip_y = y + drip_length * math.sin(angle + math.pi / 2)
                drip_points.append((drip_x, drip_y))

        def bezier_curve(points, control_points):
            if len(points) < 3 or not control_points:
                return points
            result = []
            for i in range(len(points) - 1):
                p0, p1, p2 = points[i], control_points[i], points[(i + 1) % len(points)]
                for t in range(100):
                    t /= 100
                    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
                    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
                    result.append((x, y))
            return result

        if line_type == "curved inwards":
            bubble_points = bezier_curve(bubble_points, control_points_inwards)
        elif line_type == "curved outwards":
            bubble_points = bezier_curve(bubble_points, control_points_outwards)
        elif line_type == "splat":
            bubble_points = bezier_curve(bubble_points, splat_control_points)
        elif line_type == "drip":
            for i in range(len(bubble_points)):
                bubble_points.insert(i * 2 + 1, drip_points[i])

        fill_color = self.get_rgb(bubble_fill_colour) if bubble_fill else None
        draw.polygon(bubble_points, fill=fill_color, outline=self.get_rgb(bubble_colour), width=bubble_stroke_thickness)

        # Draw a random number of shapes between 3 and 6, equally spaced around the bubble
        if shape_type != "none":
            num_shapes = random.randint(3, 6)
            step = len(bubble_points) // num_shapes
            for i in range(num_shapes):
                point = bubble_points[i * step]
                self.draw_shape(draw, shape_type, point, jagged_points // 3)

    def randomize_colours(self):
        return random.choice(list(self._available_colours.values()))

    def draw_text(self, image, mask, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours):
        self._loaded_font = self.load_font(font, font_size)
        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)
        
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

        left, top, right, bottom = mask_draw.textbbox((0, 0), self._full_text, font=self._loaded_font, stroke_width=stroke_thickness, align=horizontal_alignment)

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

        self.draw_text_with_letter_spacing(mask_draw, (self._x, self._y), self._full_text, self._loaded_font, 255, 255, stroke_thickness, horizontal_alignment, vertical_randomness, letter_spacing, justification)

        # Calculate the bounding box of the text on the mask
        mask_bbox = mask.getbbox()

        if bubble > 0 and mask_bbox:
            self.draw_bubble(draw, mask_bbox, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance)

        self.draw_text_with_letter_spacing(draw, (self._x, self._y), self._full_text, self._loaded_font, self.hex_to_rgb(fill_colour), self.hex_to_rgb(stroke_colour), stroke_thickness, horizontal_alignment, vertical_randomness, letter_spacing, justification)

        return image, mask

    def batch_process(self, image, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours):
        pbar = ProgressBar(len(image))
        if len(image.shape) == 3:
            image_np = image.cpu().numpy()
            image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
            mask = Image.new("L", image.size, 0)  # Create a new mask image
            if seed == 0:
                seed = random.randint(10**14, 10**15 - 1)
            random.seed(seed)
            if randomize_colours:
                fill_colour = self.randomize_colours()
                stroke_colour = self.randomize_colours()
                bubble_colour = self.randomize_colours()
                bubble_fill_colour = self.randomize_colours()
            image, mask = self.draw_text(image, mask, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance, seed, vertical_randomness, letter_spacing, randomize_colours)
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            mask_tensor_out = torch.tensor(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
            pbar.update_absolute(0)
            return image_tensor_out, mask_tensor_out
        else:
            image_np = image.cpu().numpy()
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_np]
            masks = [Image.new("L", img.size, 0) for img in images]  # Create masks for each image in the batch
            images_out = []
            masks_out = []
            for idx, (img, mask) in enumerate(zip(images, masks)):
                if seed == 0:
                    seed = random.randint(10**14, 10**15 - 1)
                random.seed(seed + idx)  # Different seed for each image in the batch
                if randomize_colours:
                    fill_colour = self.randomize_colours()
                    stroke_colour = self.randomize_colours()
                    bubble_colour = self.randomize_colours()
                    bubble_fill_colour = self.randomize_colours()
                img, mask = self.draw_text(img, mask, text, randomize, font_size, font, fill_colour, stroke_colour, stroke_thickness, padding, horizontal_alignment, vertical_alignment, justification, bubble, bubble_distance, bubble_colour, bubble_stroke_thickness, bubble_fill, bubble_fill_colour, line_type, shape_type, jagged_points, jagged_min_distance, jagged_max_distance, seed + idx, vertical_randomness, letter_spacing, randomize_colours)
                images_out.append(np.array(img).astype(np.float32) / 255.0)
                masks_out.append(np.array(mask).astype(np.float32) / 255.0)
                pbar.update_absolute(idx)
            images_tensor = torch.from_numpy(np.stack(images_out))
            masks_tensor = torch.from_numpy(np.stack(masks_out))
            return images_tensor, masks_tensor
