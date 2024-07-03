import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image, ImageOps
from comfy.utils import ProgressBar
import io
from datetime import datetime, timedelta
import torch

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

def save_frames_as_images(frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = []
    for i, frame in enumerate(frames):
        image_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
        cv2.imwrite(image_path, frame_rgb)
        image_paths.append(image_path)
    
    return image_paths

class GRCounterVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "finish": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                "countdown": ("BOOLEAN", {"default": False}),
                "clock_type": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "h:\\counter\\counter_video.mp4"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "font_path": (
                    cls._available_fonts,
                    {"default": cls._available_fonts[0] if cls._available_fonts else "arial.ttf"},
                ),
                "font_size_min": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 1}),
                "font_size_max": ("INT", {"default": 200, "min": 10, "max": 1000, "step": 1}),
                "font_color": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "white"},
                ),
                "font_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "font_control": (
                    ["decrease", "constant", "increase"],
                    {"default": "constant"}
                ),
                "outline": ("BOOLEAN", {"default": False}),
                "outline_size": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "outline_color": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "outline_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "resolution": (
                    list(sorted(cls._available_resolutions.keys())),
                    {"default": "HD 1280x720"},
                ),
                "counter_duration": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "processing_device": (
                    ["cpu", "gpu"],
                    {"default": "cpu"}
                ),
                "start_x": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1}),
                "start_y": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1}),
                "end_x": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1}),
                "end_y": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1}),
                "movement": (
                    ["move left", "constant", "move right"],
                    {"default": "constant"}
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("FileDetails", "FrameImages")
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
        return GRCounterVideo._available_fonts

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def hex_to_rgba(hex_color, alpha):
        r, g, b = GRCounterVideo.hex_to_rgb(hex_color)
        return (r, g, b, int(alpha * 255 / 100))

    @staticmethod
    def calculate_font_size(frame_index, total_frames, font_size_min, font_size_max, font_control):
        if font_control == "increase":
            return int(font_size_min + (font_size_max - font_size_min) * (frame_index / total_frames))
        elif font_control == "decrease":
            return int(font_size_max - (font_size_max - font_size_min) * (frame_index / total_frames))
        else:  # constant
            return font_size_min

    @staticmethod
    def calculate_position(frame_index, total_frames, start_pos, end_pos, movement):
        if movement == "constant":
            return start_pos
        elif movement == "move left":
            return int(start_pos + (end_pos - start_pos) * (frame_index / total_frames))
        elif movement == "move right":
            return int(end_pos - (end_pos - start_pos) * (frame_index / total_frames))

    @staticmethod
    def generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_control, outline, outline_size, outline_color, outline_opacity, resolution, counter_duration, fps, device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant"):
        frames = []
        frames_per_counter = int((counter_duration / 1000) * fps)
        total_frames = frames_per_counter * (abs(finish - start) + 1)
        pbar = ProgressBar(total_frames)

        if clock_type:
            total_seconds = abs(finish - start)
            if countdown:
                total_seconds = finish - start
            for idx in range(total_seconds + 1):
                seconds = start + idx if not countdown else finish - idx
                time_str = str(timedelta(seconds=seconds))
                time_str = time_str.zfill(8)  # Ensure it matches HH:MM:SS format
                for _ in range(frames_per_counter):
                    font_size = GRCounterVideo.calculate_font_size(idx, total_frames, font_size_min, font_size_max, font_control)
                    pos_x = GRCounterVideo.calculate_position(idx, total_frames, start_x, end_x, movement)
                    pos_y = GRCounterVideo.calculate_position(idx, total_frames, start_y, end_y, movement)
                    frame = GRCounterVideo.create_frame(time_str, font_path, font_size, font_color, font_opacity, outline, outline_size, outline_color, outline_opacity, resolution, device, pos_x, pos_y)
                    frames.append(frame)
                    pbar.update_absolute(idx)
                    idx += 1
        else:
            idx = 0
            if countdown:
                start, finish = finish, start
                for i in range(start, finish - 1, -1):
                    for _ in range(frames_per_counter):
                        font_size = GRCounterVideo.calculate_font_size(idx, total_frames, font_size_min, font_size_max, font_control)
                        pos_x = GRCounterVideo.calculate_position(idx, total_frames, start_x, end_x, movement)
                        pos_y = GRCounterVideo.calculate_position(idx, total_frames, start_y, end_y, movement)
                        frame = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, outline, outline_size, outline_color, outline_opacity, resolution, device, pos_x, pos_y)
                        frames.append(frame)
                        pbar.update_absolute(idx)
                        idx += 1
            else:
                for i in range(start, finish + 1):
                    for _ in range(frames_per_counter):
                        font_size = GRCounterVideo.calculate_font_size(idx, total_frames, font_size_min, font_size_max, font_control)
                        pos_x = GRCounterVideo.calculate_position(idx, total_frames, start_x, end_x, movement)
                        pos_y = GRCounterVideo.calculate_position(idx, total_frames, start_y, end_y, movement)
                        frame = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, outline, outline_size, outline_color, outline_opacity, resolution, device, pos_x, pos_y)
                        frames.append(frame)
                        pbar.update_absolute(idx)
                        idx += 1

        return frames

    @staticmethod
    def create_frame(number: str, font_path: str, font_size: int, font_color: str, font_opacity: int, outline: bool, outline_size: int, outline_color: str, outline_opacity: int, resolution: str, device="cpu", pos_x=0, pos_y=0):
        width, height = GRCounterVideo._available_resolutions[resolution]

        frame = np.zeros((height, width, 4), dtype=np.uint8)  # Use 4 channels for RGBA with transparency

        font_color = GRCounterVideo._available_colours.get(font_color.lower(), "#FFFFFF")
        font_color = GRCounterVideo.hex_to_rgba(font_color, font_opacity)

        pil_frame = Image.fromarray(frame, 'RGBA')
        draw = ImageDraw.Draw(pil_frame, 'RGBA')

        font = ImageFont.truetype(font_path, font_size) if font_path in GRCounterVideo._available_fonts else ImageFont.load_default()

        text = str(number)
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = pos_x if pos_x else (frame.shape[1] - text_width) // 2
        text_y = pos_y if pos_y else (frame.shape[0] - text_height) // 2

        if outline:
            outline_color = GRCounterVideo._available_colours.get(outline_color.lower(), "#000000")
            outline_color = GRCounterVideo.hex_to_rgba(outline_color, outline_opacity)
            for dx in range(-outline_size, outline_size + 1):
                for dy in range(-outline_size, outline_size + 1):
                    if dx**2 + dy**2 <= outline_size**2:
                        draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)

        draw.text((text_x, text_y), text, font=font, fill=font_color)
        frame = np.array(pil_frame)  # Convert back to RGBA

        return frame

    @staticmethod
    def get_file_details(file_path, start, finish, countdown, clock_type, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_control, outline, outline_size, outline_color, outline_opacity, resolution, counter_duration, start_x, start_y, end_x, end_y, movement):
        file_size = os.path.getsize(file_path)
        file_details = (
            f"File Path: {file_path}, File Size: {file_size} bytes\n"
            f"Start: {start}, Finish: {finish}, Countdown: {countdown}, Clock Type: {clock_type}, FPS: {fps}\n"
            f"Font Path: {font_path}, Font Size Min: {font_size_min}, Font Size Max: {font_size_max}, Font Color: {font_color}, Font Opacity: {font_opacity}\n"
            f"Font Control: {font_control}\n"
            f"Outline: {outline}, Outline Size: {outline_size}, Outline Color: {outline_color}, Outline Opacity: {outline_opacity}\n"
            f"Resolution: {resolution}, Width: {GRCounterVideo._available_resolutions[resolution][0]}, Height: {GRCounterVideo._available_resolutions[resolution][1]}\n"
            f"Colour Mode: RGBA, Channels: 4, Counter Duration: {counter_duration} ms\n"
            f"Start X: {start_x}, Start Y: {start_y}, End X: {end_x}, End Y: {end_y}, Movement: {movement}"
        )
        return file_details

    @classmethod
    def generate_video(cls, start, finish, countdown, clock_type, output_path, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_control, outline, outline_size, outline_color, outline_opacity, resolution, counter_duration, processing_device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant"):
        # Append date and time to the output path
        now = datetime.now().strftime("%d-%y-%m-%H-%M")
        output_path = os.path.splitext(output_path)[0] + f"_{now}.mp4"
        
        frames = cls.generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_control, outline, outline_size, outline_color, outline_opacity, resolution, counter_duration, fps, device=processing_device, start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y, movement=movement)
        output_file = save_video(frames, output_path, fps)
        file_details = cls.get_file_details(output_file, start, finish, countdown, clock_type, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_control, outline, outline_size, outline_color, outline_opacity, resolution, counter_duration, start_x, start_y, end_x, end_y, movement)
        
        # Ensure frames are in the correct format before converting to tensors
        frame_images = [torch.from_numpy(frame).permute(0, 1, 2) for frame in frames]

        return (file_details, frame_images)
