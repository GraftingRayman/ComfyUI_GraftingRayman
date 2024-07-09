import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from datetime import datetime
import torch
from comfy.utils import ProgressBar, common_upscale  # Ensure these modules are available in your environment
import random

def save_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR
        out.write(frame_bgr)

    out.release()
    return output_path

def save_frames_as_images(frames, output_folder, timestamp):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = []
    for i, frame in enumerate(frames):
        image_path = os.path.join(output_folder, f"frame_{i:04d}_{timestamp}.png")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR
        cv2.imwrite(image_path, frame_rgb)
        image_paths.append(image_path)
    
    return image_paths

def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] or [channels, height, width] or [height, width, channels] to PIL Image
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[batch_index]
    img_np = 255. * img_tensor.cpu().numpy()

    print(f"tensor_to_pil: img_np shape = {img_np.shape}")

    if img_np.ndim == 3:
        if img_np.shape[0] == 1:  # Grayscale
            img_np = img_np.squeeze(0)
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='L')
        elif img_np.shape[0] == 3:  # RGB
            img_np = np.moveaxis(img_np, 0, -1)
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='RGB')
        elif img_np.shape[0] == 4:  # RGBA
            img_np = np.moveaxis(img_np, 0, -1)
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='RGBA')
        elif img_np.shape[2] == 3:  # RGB [height, width, channels]
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='RGB')
        elif img_np.shape[2] == 4:  # RGBA [height, width, channels]
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels in the input tensor: {img_np.shape}")
    else:
        raise ValueError(f"Unsupported tensor shape: {img_np.shape}")
    
    return img

class GRScrollerVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Sample Text"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "resolution": (
                    list(sorted(cls._available_resolutions.keys())),
                    {"default": "HD 1280x720"},
                ),
                "use_background_image_size": ("BOOLEAN", {"default": False}),
                "use_background_image": ("BOOLEAN", {"default": False}),
                "font_path": (
                    cls._available_fonts,
                    {"default": cls._available_fonts[0] if cls._available_fonts else "arial.ttf"},
                ),
                "font_size": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 1}),
                "font_color": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "white"},
                ),
                "font_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "font_stroke": ("BOOLEAN", {"default": False}),
                "font_stroke_thickness": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "font_stroke_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "background_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "output_path": ("STRING", {"default": "H:\\scroller\\output_video.mp4"}),
                "save_images": ("BOOLEAN", {"default": False}),
                "image_save_path": ("STRING", {"default": "H:\\scroller\\frames\\"}),
                "scroll_direction": (
                    ["left", "right", "up", "down"],
                    {"default": "left"},
                ),
                "scroll_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 250.0, "step": 0.1}),
                "movement_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "movement_duration": ("INT", {"default": 10, "min": 1, "max": 600, "step": 1}),
                "movement_type": (
                    ["scroll", "bounce", "wave", "jitter", "circular", "zoom", "fade", "rotate", "oscillate", "diagonal", "spiral", "flip", "random_walk", "shake", "pulse", "sway", "grow_shrink", "blur", "fade_in_out", "rotate_3d", "flip_horizontal", "flip_vertical", "color_change", "ripple", "glitch"],
                    {"default": "scroll"},
                ),
                "movement_distance": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "apply_to_each_character": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": random.randint(0, 2**32 - 1), "min": 0, "max": 2**32 - 1}),
                "upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "background_image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("FileDetails", "Frames")
    CATEGORY = "GraftingRayman/Video"
    FUNCTION = "generate_video"

    def __init__(self):
        pass

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
        "black": "#000000",
        "white": "#FFFFFF",
        "red": "#FF0000",
        "green": "#008000",
        "blue": "#0000FF",
        "yellow": "#FFFF00",
        "cyan": "#00FFFF",
        "magenta": "#FF00FF",
        "gray": "#808080",
        "silver": "#C0C0C0",
        "maroon": "#800000",
        "olive": "#808000",
        "purple": "#800080",
        "teal": "#008080",
        "navy": "#000080"
    }

    _available_resolutions = {
        "SD 640x480": (640, 480),
        "HD 1280x720": (1280, 720),
        "Full HD 1920x1080": (1920, 1080),
        "4K 3840x2160": (3840, 2160),
        "Portrait HD 720x1280": (720, 1280),
        "Portrait Full HD 1080x1920": (1080, 1920),
        "Portrait 4K 2160x3840": (2160, 3840),
    }

    @staticmethod
    def get_available_fonts():
        return GRScrollerVideo._available_fonts

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def hex_to_rgba(hex_color, alpha):
        r, g, b = GRScrollerVideo.hex_to_rgb(hex_color)
        return (r, g, b, int(alpha * 255 / 100))

    @staticmethod
    def draw_text(draw, position, text, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour):
        if font_stroke:
            # Draw the stroke outline
            stroke_color = GRScrollerVideo._available_colours.get(font_stroke_colour.lower(), "#000000")
            stroke_color = GRScrollerVideo.hex_to_rgba(stroke_color, 100)
            x, y = position
            for dx in range(-font_stroke_thickness, font_stroke_thickness + 1):
                for dy in range(-font_stroke_thickness, font_stroke_thickness + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
        
        # Draw the main text
        draw.text(position, text, font=font, fill=font_color)

    @staticmethod
    def generate_frames(text, font_path, font_size, font_color, font_opacity, background_colour, use_background_image_size, use_background_image, background_image, resolution, scroll_direction, scroll_speed, movement_speed, movement_duration, fps, movement_type, movement_distance, apply_to_each_character, seed, font_stroke, font_stroke_thickness, font_stroke_colour, upscale):
        frames = []
        foreground_frames = []
        width, height = GRScrollerVideo._available_resolutions[resolution]

        if use_background_image_size and background_image is not None:
            if isinstance(background_image, torch.Tensor):
                background_image = tensor_to_pil(background_image)
            width, height = background_image.size
        
        total_frames = movement_duration * fps
        
        background_color = GRScrollerVideo._available_colours.get(background_colour.lower(), "#000000")
        background_color = GRScrollerVideo.hex_to_rgba(background_color, 100)
        
        if use_background_image and background_image is not None:
            if isinstance(background_image, torch.Tensor):
                background_image = tensor_to_pil(background_image)
            
            # Upscale the background image first
            upscale_background = int(upscale)
            if upscale_background > 1:
                background_image = background_image.resize((background_image.width * upscale_background, background_image.height * upscale_background), Image.LANCZOS)
            
            pil_background = background_image.resize((width, height)).convert("RGBA")
        else:
            pil_background = Image.new('RGBA', (width, height), background_color)

        draw = ImageDraw.Draw(pil_background, 'RGBA')
        font_color = GRScrollerVideo._available_colours.get(font_color.lower(), "#FFFFFF")
        font_color = GRScrollerVideo.hex_to_rgba(font_color, font_opacity)
        font = ImageFont.truetype(font_path, font_size) if font_path in GRScrollerVideo._available_fonts else ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        pbar = ProgressBar(total_frames)  # Initialize the progress bar

        # Ensure the seed is within the valid range
        seed = seed % 2**32

        # Set the random seed
        np.random.seed(seed)
        random.seed(seed)

        # Choose random movement types for each character if apply_to_each_character is True
        if apply_to_each_character:
            movement_types = ["bounce", "wave", "jitter", "circular", "zoom", "fade", "rotate", "oscillate", "diagonal", "spiral", "flip", "random_walk", "shake", "pulse", "sway", "grow_shrink", "blur", "fade_in_out", "rotate_3d", "flip_horizontal", "flip_vertical", "color_change", "ripple", "glitch"]
            char_movements = [random.choice(movement_types) for _ in text]

        for frame_index in range(total_frames):
            frame = pil_background.copy()
            foreground_frame = Image.new('RGBA', (width, height))
            draw_foreground = ImageDraw.Draw(foreground_frame, 'RGBA')
            
            if movement_type == "scroll":
                if scroll_direction == "left":
                    x_position = int(width - (frame_index * scroll_speed) % (text_width + width))
                    y_position = (height - text_height) // 2
                elif scroll_direction == "right":
                    x_position = int(-text_width + (frame_index * scroll_speed) % (text_width + width))
                    y_position = (height - text_height) // 2
                elif scroll_direction == "up":
                    x_position = (width - text_width) // 2
                    y_position = int(height - (frame_index * scroll_speed) % (text_height + height))
                elif scroll_direction == "down":
                    x_position = (width - text_width) // 2
                    y_position = int(-text_height + (frame_index * scroll_speed) % (text_height + height))
                GRScrollerVideo.draw_text(draw_foreground, (x_position, y_position), text, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
            
            else:
                for i, char in enumerate(text):
                    char_x = (width - text_width) // 2 + i * font_size
                    char_y = (height - text_height) // 2
                    
                    offset_x = 0
                    offset_y = 0
                    char_image = None

                    if apply_to_each_character:
                        char_movement_type = char_movements[i]
                    else:
                        char_movement_type = movement_type
                    
                    if char_movement_type == "bounce":
                        offset_y = int(movement_distance * np.sin(2 * np.pi * frame_index * movement_speed / fps))
                    elif char_movement_type == "wave":
                        offset_y = int(movement_distance * np.sin(2 * np.pi * (frame_index + i) * movement_speed / fps))
                    elif char_movement_type == "jitter":
                        offset_x = random.randint(-movement_distance, movement_distance)
                        offset_y = random.randint(-movement_distance, movement_distance)
                    elif char_movement_type == "circular":
                        radius = min(width, height) // 4
                        angle = 2 * np.pi * (frame_index * movement_speed / fps + i / len(text))
                        offset_x = int(radius * np.cos(angle))
                        offset_y = int(radius * np.sin(angle))
                    elif char_movement_type == "zoom":
                        scale = 1 + 0.5 * np.sin(2 * np.pi * frame_index * movement_speed / fps)
                        char_x = int(char_x * scale)
                        char_y = int(char_y * scale)
                    elif char_movement_type == "fade":
                        opacity = int(255 * (0.5 * (1 + np.sin(2 * np.pi * frame_index * movement_speed / fps))))
                        temp_font_color = (font_color[0], font_color[1], font_color[2], opacity)
                        GRScrollerVideo.draw_text(draw_foreground, (char_x + offset_x, char_y + offset_y), char, font, temp_font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        continue  # Move to the next character or frame
                    elif char_movement_type == "rotate":
                        angle = 360 * frame_index * movement_speed / total_frames
                        rotated_char = Image.new('RGBA', (width, height))
                        temp_draw = ImageDraw.Draw(rotated_char)
                        GRScrollerVideo.draw_text(temp_draw, (char_x, char_y), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        rotated_char = rotated_char.rotate(angle, expand=1)
                        frame.paste(rotated_char, (0, 0), rotated_char)
                        continue
                    elif char_movement_type == "oscillate":
                        offset_x = int(movement_distance * np.sin(2 * np.pi * frame_index * movement_speed / fps))
                    elif char_movement_type == "diagonal":
                        offset_x = int(frame_index * scroll_speed * movement_speed) % (width + text_width) - text_width
                        offset_y = int(frame_index * scroll_speed * movement_speed) % (height + text_height) - text_height
                    elif char_movement_type == "spiral":
                        radius = min(width, height) // 4 * (frame_index * movement_speed / total_frames)
                        angle = 2 * np.pi * frame_index * movement_speed / fps
                        offset_x = int(radius * np.cos(angle))
                        offset_y = int(radius * np.sin(angle))
                    elif char_movement_type == "flip":
                        char_image = Image.new('RGBA', (font_size, font_size))
                        temp_draw = ImageDraw.Draw(char_image)
                        GRScrollerVideo.draw_text(temp_draw, (0, 0), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        char_image = char_image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif char_movement_type == "random_walk":
                        offset_x = random.randint(-movement_distance, movement_distance)
                        offset_y = random.randint(-movement_distance, movement_distance)
                    elif char_movement_type == "shake":
                        offset_x = random.randint(-movement_distance, movement_distance)
                        offset_y = random.randint(-movement_distance, movement_distance)
                    elif char_movement_type == "pulse":
                        scale = 1 + 0.5 * np.sin(2 * np.pi * frame_index * movement_speed / fps)
                        char_x = int(char_x * scale)
                        char_y = int(char_y * scale)
                    elif char_movement_type == "sway":
                        offset_x = int(movement_distance * np.sin(2 * np.pi * frame_index * movement_speed / fps))
                    elif char_movement_type == "grow_shrink":
                        scale = 1 + 0.5 * np.sin(2 * np.pi * frame_index * movement_speed / fps)
                        char_x = int(char_x * scale)
                        char_y = int(char_y * scale)
                    elif char_movement_type == "blur":
                        blur_radius = int(5 * np.sin(2 * np.pi * frame_index * movement_speed / fps))
                        char_image = Image.new('RGBA', (font_size, font_size))
                        temp_draw = ImageDraw.Draw(char_image)
                        GRScrollerVideo.draw_text(temp_draw, (0, 0), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        char_image = char_image.filter(ImageFilter.GaussianBlur(blur_radius))
                    elif char_movement_type == "fade_in_out":
                        opacity = int(255 * (0.5 * (1 + np.sin(2 * np.pi * frame_index * movement_speed / fps))))
                        temp_font_color = (font_color[0], font_color[1], font_color[2], opacity)
                        GRScrollerVideo.draw_text(draw_foreground, (char_x + offset_x, char_y + offset_y), char, font, temp_font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        continue
                    elif char_movement_type == "rotate_3d":
                        angle = 360 * frame_index * movement_speed / total_frames
                        rotated_char = Image.new('RGBA', (font_size, font_size))
                        temp_draw = ImageDraw.Draw(rotated_char)
                        GRScrollerVideo.draw_text(temp_draw, (0, 0), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        rotated_char = rotated_char.rotate(angle, expand=1)
                        foreground_frame.paste(rotated_char, (char_x, char_y), rotated_char)
                        continue
                    elif char_movement_type == "flip_horizontal":
                        char_image = Image.new('RGBA', (font_size, font_size))
                        temp_draw = ImageDraw.Draw(char_image)
                        GRScrollerVideo.draw_text(temp_draw, (0, 0), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        char_image = char_image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif char_movement_type == "flip_vertical":
                        char_image = Image.new('RGBA', (font_size, font_size))
                        temp_draw = ImageDraw.Draw(char_image)
                        GRScrollerVideo.draw_text(temp_draw, (0, 0), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        char_image = char_image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif char_movement_type == "color_change":
                        temp_font_color = GRScrollerVideo.hex_to_rgba(random.choice(list(GRScrollerVideo._available_colours.values())), font_opacity)
                        GRScrollerVideo.draw_text(draw_foreground, (char_x + offset_x, char_y + offset_y), char, font, temp_font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        continue
                    elif char_movement_type == "ripple":
                        offset_y = int(movement_distance * np.sin(2 * np.pi * (frame_index + i) * movement_speed / fps))
                    elif char_movement_type == "glitch":
                        offset_x = random.randint(-movement_distance, movement_distance)
                        offset_y = random.randint(-movement_distance, movement_distance)
                        temp_font_color = GRScrollerVideo.hex_to_rgba(random.choice(list(GRScrollerVideo._available_colours.values())), font_opacity)
                        GRScrollerVideo.draw_text(draw_foreground, (char_x + offset_x, char_y + offset_y), char, font, temp_font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        continue

                    if apply_to_each_character:
                        if char_image:
                            foreground_frame.paste(char_image, (char_x + offset_x, char_y + offset_y), char_image)
                        else:
                            GRScrollerVideo.draw_text(draw_foreground, (char_x + offset_x, char_y + offset_y), char, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                    else:
                        x_position = char_x + offset_x
                        y_position = char_y + offset_y
                        GRScrollerVideo.draw_text(draw_foreground, (x_position, y_position), text, font, font_color, font_stroke, font_stroke_thickness, font_stroke_colour)
                        break  # Draw the whole text and break

            combined_frame = Image.alpha_composite(frame, foreground_frame)
            frames.append(np.array(combined_frame))  # Convert PIL image to numpy array before adding to the list
            foreground_frames.append(np.array(foreground_frame))  # Store foreground frame separately
            pbar.update_absolute(frame_index)  # Update the progress bar

        return frames, foreground_frames

    @staticmethod
    def get_file_details(file_path, text, font_path, font_size, font_color, font_opacity, background_colour, use_background_image_size, use_background_image, resolution, scroll_direction, scroll_speed, movement_speed, movement_duration, fps, movement_type, movement_distance, apply_to_each_character, seed, font_stroke, font_stroke_thickness, font_stroke_colour):
        file_size = os.path.getsize(file_path)
        file_details = (
            f"File Path: {file_path}, File Size: {file_size} bytes\n"
            f"Text: {text}, Font Path: {font_path}, Font Size: {font_size}, Font Color: {font_color}, Font Opacity: {font_opacity}, Font Stroke: {font_stroke}, Font Stroke Thickness: {font_stroke_thickness}, Font Stroke Colour: {font_stroke_colour}\n"
            f"Background Colour: {background_colour}, Use Background Image Size: {use_background_image_size}, Use Background Image: {use_background_image}\n"
            f"Resolution: {resolution}, Width: {GRScrollerVideo._available_resolutions[resolution][0]}, Height: {GRScrollerVideo._available_resolutions[resolution][1]}\n"
            f"Scroll Direction: {scroll_direction}, Scroll Speed: {scroll_speed}, Movement Speed: {movement_speed}, Movement Duration: {movement_duration} seconds, FPS: {fps}\n"
            f"Movement Type: {movement_type}, Movement Distance: {movement_distance}, Apply to Each Character: {apply_to_each_character}, Random Seed: {seed}"
        )
        return file_details

    @staticmethod
    def upscale(image, upscale_method, scale_by):
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return s

    @classmethod
    def generate_video(cls, text, font_path, font_size, font_color, font_opacity, background_colour, use_background_image_size=False, use_background_image=False, background_image=None, resolution='HD 1280x720', scroll_direction='left', scroll_speed=1.0, movement_speed=1.0, movement_duration=10, output_path='H:\\scroller\\output_video.mp4', save_images=False, image_save_path='H:\\scroller\\frames\\', fps=30, movement_type='scroll', movement_distance=20, apply_to_each_character=False, seed=42, font_stroke=False, font_stroke_thickness=2, font_stroke_colour="black", upscale=1.0):
        # Append date and time to the output path
        now = datetime.now().strftime("%d-%y-%m-%H-%M-%S")
        output_path = os.path.splitext(output_path)[0] + f"_{now}.mp4"

        frames, foreground_frames = cls.generate_frames(text, font_path, font_size, font_color, font_opacity, background_colour, use_background_image_size, use_background_image, background_image, resolution, scroll_direction, scroll_speed, movement_speed, movement_duration, fps, movement_type, movement_distance, apply_to_each_character, seed, font_stroke, font_stroke_thickness, font_stroke_colour, upscale)
        
        # Save video using the combined frames
        output_file = save_video(frames, output_path, fps)

        image_paths = []

        # Save frames as images if save_images is True
        if save_images:
            image_paths = save_frames_as_images(frames, image_save_path, now)

        # Convert foreground frames to tensors and stack them into a batch
        frame_tensors = [torch.from_numpy(frame).permute(0, 1, 2) for frame in foreground_frames]  # Convert to (C, H, W) format
        batch_tensor = torch.stack(frame_tensors)  # Create a batch in (N, C, H, W) format

        # Upscale the batch tensor using the upscale method if upscale >= 1.0
        if upscale >= 1.0:
            upscaled_batch_tensor = cls.upscale(batch_tensor, "lanczos", upscale)
        else:
            upscaled_batch_tensor = cls.upscale(batch_tensor, "lanczos", upscale)

        file_details = cls.get_file_details(output_file, text, font_path, font_size, font_color, font_opacity, background_colour, use_background_image_size, use_background_image, resolution, scroll_direction, scroll_speed, movement_speed, movement_duration, fps, movement_type, movement_distance, apply_to_each_character, seed, font_stroke, font_stroke_thickness, font_stroke_colour)

        return (file_details, upscaled_batch_tensor)
