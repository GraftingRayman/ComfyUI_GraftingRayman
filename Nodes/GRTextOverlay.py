import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

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
    CATEGORY = "image/text"

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
