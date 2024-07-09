import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image, ImageOps
from datetime import datetime, timedelta
import torch
from comfy.utils import ProgressBar, common_upscale  # Ensure this module is available in your environment

# Conversion function
def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] or [channels, height, width] or [height, width, channels] to PIL Image
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[batch_index]
    img_np = 255. * img_tensor.cpu().numpy()

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

def save_video(frames, output_path, fps, background_colour):
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

class GRCounterVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "finish": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                "counter_duration": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "resolution": (
                    list(sorted(cls._available_resolutions.keys())),
                    {"default": "HD 1280x720"},
                ),
                "use_background_image_size": ("BOOLEAN", {"default": False}),
                "use_background_image": ("BOOLEAN", {"default": False}),
                "countdown": ("BOOLEAN", {"default": False}),
                "clock_type": ("BOOLEAN", {"default": False}),
                "video_path": ("STRING", {"default": "h:\\counter\\counter_video.mp4"}),
                "save_images": ("BOOLEAN", {"default": False}),
                "image_output_folder": ("STRING", {"default": "h:\\counter\\images\\"}),
                "background_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
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
                "font_shadow": ("BOOLEAN", {"default": False}),
                "font_shadow_dist": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "font_control": (
                    ["decrease", "constant", "increase", "pulse"],
                    {"default": "constant"}
                ),
                "outline": ("BOOLEAN", {"default": False}),
                "outline_size": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "outline_color": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "outline_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "rotate": ("BOOLEAN", {"default": False}),
                "rotate_type": (
                    ["clockwise", "anticlockwise", "flip", "rocker", "rockerflip"],
                    {"default": "clockwise"}
                ),
                "rotate_freq": ("FLOAT", {"default": 1, "min": 0.01, "max": 50, "step": 0.01}),
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
                "upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "background_image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("FileDetails", "FrameImages", "UpscaledImage")
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
    def calculate_font_size(frame_index, total_frames, font_size_min, font_size_max, font_control, frames_per_counter=None):
        if font_control == "increase":
            return int(font_size_min + (font_size_max - font_size_min) * (frame_index / total_frames))
        elif font_control == "decrease":
            return int(font_size_max - (font_size_max - font_size_min) * (frame_index / total_frames))
        elif font_control == "pulse" and frames_per_counter is not None:
            cycle_position = frame_index % frames_per_counter
            return int(font_size_min + (font_size_max - font_size_min) * (cycle_position / frames_per_counter))
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
    def rotate_text_image(image, angle, background_size):
        rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
        # Create a new image with the original background size
        final_image = Image.new('RGBA', background_size, (0, 0, 0, 0))
        # Paste the rotated text onto the final image, centered
        final_image.paste(rotated_image, ((background_size[0] - rotated_image.size[0]) // 2, (background_size[1] - rotated_image.size[1]) // 2), rotated_image)
        return final_image

    @staticmethod
    def perspective_transform_y_axis(image, angle, width, height):
        angle_rad = np.deg2rad(angle)
        a = np.cos(angle_rad)
        b = np.sin(angle_rad)
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        dst_points = np.float32([
            [width * (1 - a) / 2, height * (1 - b) / 2],
            [width * (1 + a) / 2, height * (1 - b) / 2],
            [width * (1 + a) / 2, height * (1 + b) / 2],
            [width * (1 - a) / 2, height * (1 + b) / 2]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(np.array(image), matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return Image.fromarray(transformed_image)

    @staticmethod
    def generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, use_background_image_size, counter_duration, fps, rotate, rotate_type, rotate_freq, device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant"):
        frames_with_bg = []
        frames_without_bg = []
        frames_per_counter = int((counter_duration / 1000) * fps)
        total_frames = frames_per_counter * (abs(finish - start) + 1)

        pbar = ProgressBar(total_frames)  # Initialize the progress bar

        if use_background_image_size and background_image is not None:
            if isinstance(background_image, torch.Tensor):
                background_image = tensor_to_pil(background_image)
            width, height = background_image.size
        else:
            width, height = GRCounterVideo._available_resolutions[resolution]

        if clock_type:
            total_seconds = abs(finish - start)
            if countdown:
                total_seconds = finish - start
            for idx in range(total_seconds + 1):
                seconds = start + idx if not countdown else finish - idx
                time_str = str(timedelta(seconds=seconds))
                time_str = time_str.zfill(8)  # Ensure it matches HH:MM:SS format
                for frame_idx in range(frames_per_counter):
                    overall_frame_idx = idx * frames_per_counter + frame_idx
                    font_size = GRCounterVideo.calculate_font_size(
                        overall_frame_idx if font_control != "pulse" else frame_idx,
                        total_frames,
                        font_size_min,
                        font_size_max,
                        font_control,
                        frames_per_counter
                    )
                    pos_x = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_x, end_x, movement)
                    pos_y = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_y, end_y, movement)
                    frame_with_bg, frame_without_bg = GRCounterVideo.create_frame(time_str, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, (width, height), device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                    
                    frames_with_bg.append(frame_with_bg)
                    frames_without_bg.append(frame_without_bg)
                    pbar.update_absolute(overall_frame_idx)  # Update the progress bar
        else:
            idx = 0
            if countdown:
                start, finish = finish, start
                for i in range(start, finish - 1, -1):
                    for frame_idx in range(frames_per_counter):
                        overall_frame_idx = idx * frames_per_counter + frame_idx
                        font_size = GRCounterVideo.calculate_font_size(
                            overall_frame_idx if font_control != "pulse" else frame_idx,
                            total_frames,
                            font_size_min,
                            font_size_max,
                            font_control,
                            frames_per_counter
                        )
                        pos_x = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_x, end_x, movement)
                        pos_y = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_y, end_y, movement)
                        frame_with_bg, frame_without_bg = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, (width, height), device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                        
                        frames_with_bg.append(frame_with_bg)
                        frames_without_bg.append(frame_without_bg)
                        pbar.update_absolute(overall_frame_idx)  # Update the progress bar
                    idx += 1
            else:
                for i in range(start, finish + 1):
                    for frame_idx in range(frames_per_counter):
                        overall_frame_idx = idx * frames_per_counter + frame_idx
                        font_size = GRCounterVideo.calculate_font_size(
                            overall_frame_idx if font_control != "pulse" else frame_idx,
                            total_frames,
                            font_size_min,
                            font_size_max,
                            font_control,
                            frames_per_counter
                        )
                        pos_x = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_x, end_x, movement)
                        pos_y = GRCounterVideo.calculate_position(overall_frame_idx, total_frames, start_y, end_y, movement)
                        frame_with_bg, frame_without_bg = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, (width, height), device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                        
                        frames_with_bg.append(frame_with_bg)
                        frames_without_bg.append(frame_without_bg)
                        pbar.update_absolute(overall_frame_idx)  # Update the progress bar
                    idx += 1

        return frames_with_bg, frames_without_bg

    @staticmethod
    def create_frame(number: str, font_path: str, font_size: int, font_color: str, font_opacity: int, font_shadow: bool, font_shadow_dist: int, outline: bool, outline_size: int, outline_color: str, outline_opacity: int, background_colour: str, use_background_image: bool, background_image, resolution: tuple, device="cpu", pos_x=0, pos_y=0, rotate=False, rotate_type="clockwise", rotate_freq=1, frame_idx=0, frames_per_counter=1):
        width, height = resolution

        if use_background_image and background_image is not None:
            if isinstance(background_image, torch.Tensor):
                background_image = tensor_to_pil(background_image)
            background_image = background_image.resize((width, height)).convert("RGBA")
            
            # Upscale the background image before using it
            upscale_factor = 2  # Example upscale factor, you can adjust as needed
            new_width, new_height = int(width * upscale_factor), int(height * upscale_factor)
            background_image = background_image.resize((new_width, new_height), Image.LANCZOS)
            
            pil_frame_with_bg = background_image.resize((width, height)).convert("RGBA")
        else:
            background_color = GRCounterVideo._available_colours.get(background_colour.lower(), "#000000")
            background_color = GRCounterVideo.hex_to_rgba(background_color, 100)
            pil_frame_with_bg = Image.new('RGBA', (width, height), background_color)

        pil_frame_without_bg = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        font_color = GRCounterVideo._available_colours.get(font_color.lower(), "#FFFFFF")
        font_color = GRCounterVideo.hex_to_rgba(font_color, font_opacity)

        font = ImageFont.truetype(font_path, font_size) if font_path in GRCounterVideo._available_fonts else ImageFont.load_default()

        text = str(number)
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = pos_x if pos_x else (width - text_width) // 2
        text_y = pos_y if pos_y else (height - text_height) // 2

        # Create a transparent layer for the text and its shadow
        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer, 'RGBA')

        if font_shadow:
            shadow_color = (0, 0, 0, int(font_opacity * 255 / 100))
            shadow_x = text_x + font_shadow_dist
            shadow_y = text_y + font_shadow_dist
            text_draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)

        if outline:
            outline_color = GRCounterVideo._available_colours.get(outline_color.lower(), "#000000")
            outline_color = GRCounterVideo.hex_to_rgba(outline_color, outline_opacity)
            for dx in range(-outline_size, outline_size + 1):
                for dy in range(-outline_size, outline_size + 1):
                    if dx**2 + dy**2 <= outline_size**2:
                        text_draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)

        text_draw.text((text_x, text_y), text, font=font, fill=font_color)

        if rotate:
            rotate_phase = (frame_idx / frames_per_counter) % 1
            full_rotations = frame_idx // frames_per_counter
            if rotate_type == "clockwise":
                angle = 360 * rotate_freq * rotate_phase
            elif rotate_type == "anticlockwise":
                angle = -360 * rotate_freq * rotate_phase
            elif rotate_type == "flip":
                angle = 30 * np.sin(2 * np.pi * rotate_freq * rotate_phase)
                text_layer = GRCounterVideo.perspective_transform_y_axis(text_layer, angle, width, height)
            elif rotate_type == "rocker":
                angle = 30 * np.sin(2 * np.pi * rotate_freq * rotate_phase)
            elif rotate_type == "rockerflip":
                angle = 30 * np.sin(2 * np.pi * rotate_freq * rotate_phase)
                text_layer = GRCounterVideo.perspective_transform_y_axis(text_layer, angle, width, height)
            if rotate_type not in ["flip", "rockerflip"]:
                text_layer = GRCounterVideo.rotate_text_image(text_layer, angle, (width, height))

        # Composite the text layer onto the background
        pil_frame_with_bg = Image.alpha_composite(pil_frame_with_bg, text_layer)
        pil_frame_without_bg = Image.alpha_composite(pil_frame_without_bg, text_layer)

        frame_with_bg = np.array(pil_frame_with_bg)  # Convert back to RGBA
        frame_without_bg = np.array(pil_frame_without_bg)  # Convert back to RGBA

        return frame_with_bg, frame_without_bg

    @staticmethod
    def get_file_details(file_path, start, finish, countdown, clock_type, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, resolution, counter_duration, start_x, start_y, end_x, end_y, movement):
        file_size = os.path.getsize(file_path)
        file_details = (
            f"File Path: {file_path}, File Size: {file_size} bytes\n"
            f"Start: {start}, Finish: {finish}, Countdown: {countdown}, Clock Type: {clock_type}, FPS: {fps}\n"
            f"Font Path: {font_path}, Font Size Min: {font_size_min}, Font Size Max: {font_size_max}, Font Color: {font_color}, Font Opacity: {font_opacity}, Font Shadow: {font_shadow}, Font Shadow Dist: {font_shadow_dist}\n"
            f"Font Control: {font_control}\n"
            f"Outline: {outline}, Outline Size: {outline_size}, Outline Color: {outline_color}, Outline Opacity: {outline_opacity}\n"
            f"Background Colour: {background_colour}, Use Background Image: {use_background_image}\n"
            f"Resolution: {resolution}, Width: {GRCounterVideo._available_resolutions[resolution][0]}, Height: {GRCounterVideo._available_resolutions[resolution][1]}\n"
            f"Colour Mode: RGBA, Channels: 4, Counter Duration: {counter_duration} ms\n"
            f"Start X: {start_x}, Start Y: {start_y}, End X: {end_x}, End Y: {end_y}, Movement: {movement}"
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
    def generate_video(cls, start, finish, countdown, clock_type, video_path, save_images, image_output_folder, resolution, use_background_image_size, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image=False, background_image=None, counter_duration=1000, rotate=False, rotate_type="clockwise", rotate_freq=1, processing_device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant", upscale=1.0):
        # Append date and time to the output path
        now = datetime.now().strftime("%d-%y-%m-%H-%M-%S")
        video_path = os.path.splitext(video_path)[0] + f"_{now}.mp4"

        frames_with_bg, frames_without_bg = cls.generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, use_background_image_size, counter_duration, fps, rotate, rotate_type, rotate_freq, device=processing_device, start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y, movement=movement)
        output_file = save_video(frames_with_bg, video_path, fps, background_colour)
        file_details = cls.get_file_details(output_file, start, finish, countdown, clock_type, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, resolution, counter_duration, start_x, start_y, end_x, end_y, movement)
        
        image_paths = []
        if save_images:
            # Save frames as images in the specified folder
            image_paths = save_frames_as_images(frames_with_bg, image_output_folder, now)
        
        # Ensure frames are in the correct format before converting to tensors
        frame_images = [torch.from_numpy(frame).permute(0, 1, 2) for frame in frames_without_bg]  # Convert to (C, H, W) format
        batch_tensor = torch.stack(frame_images)  # Create a batch in (N, C, H, W) format

        # Upscale the batch tensor using the upscale method if upscale >= 1.0
        if upscale >= 1.0:
            upscaled_batch_tensor = cls.upscale(batch_tensor, "lanczos", upscale)
        else:
            upscaled_batch_tensor = batch_tensor

        return (file_details, upscaled_batch_tensor)
