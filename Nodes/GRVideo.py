import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image, ImageOps
from datetime import datetime, timedelta
import torch
from comfy.utils import ProgressBar, common_upscale  # Ensure this module is available in your environment

# Conversion functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2numpy(image):
    return (255.0 * image.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(0), 0, 255).astype(np.uint8))

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def rgba2rgb_tensor(rgba):
    r = rgba[..., 0]
    g = rgba[..., 1]
    b = rgba[..., 2]
    return torch.stack([r, g, b], dim=3)

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
                "background_colour": (
                    list(sorted(cls._available_colours.keys())),
                    {"default": "black"},
                ),
                "use_background_image": ("BOOLEAN", {"default": False}),
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
                "upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1})

            },
            "optional": {
                "background_image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE","IMAGE")
    RETURN_NAMES = ("FileDetails", "FrameImages","UpscaledImage")
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
    def generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, counter_duration, fps, rotate, rotate_type, rotate_freq, device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant"):
        frames = []
        frames_per_counter = int((counter_duration / 1000) * fps)
        total_frames = frames_per_counter * (abs(finish - start) + 1)

        pbar = ProgressBar(total_frames)  # Initialize the progress bar

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
                    frame = GRCounterVideo.create_frame(time_str, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                    
                    frames.append(frame)
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
                        frame = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                        
                        frames.append(frame)
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
                        frame = GRCounterVideo.create_frame(i, font_path, font_size, font_color, font_opacity, font_shadow, font_shadow_dist, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, device, pos_x, pos_y, rotate, rotate_type, rotate_freq, overall_frame_idx, frames_per_counter)
                        
                        frames.append(frame)
                        pbar.update_absolute(overall_frame_idx)  # Update the progress bar
                    idx += 1

        return frames

    @staticmethod
    def create_frame(number: str, font_path: str, font_size: int, font_color: str, font_opacity: int, font_shadow: bool, font_shadow_dist: int, outline: bool, outline_size: int, outline_color: str, outline_opacity: int, background_colour: str, use_background_image: bool, background_image, resolution: str, device="cpu", pos_x=0, pos_y=0, rotate=False, rotate_type="clockwise", rotate_freq=1, frame_idx=0, frames_per_counter=1):
        width, height = GRCounterVideo._available_resolutions[resolution]

        if use_background_image and background_image is not None:
            if isinstance(background_image, torch.Tensor):
                if background_image.dim() == 4:
                    background_image = background_image[0]  # Use the first image in the batch
                background_image = background_image.squeeze().cpu().numpy()
                if background_image.ndim == 3 and background_image.shape[0] in {1, 3, 4}:
                    background_image = np.moveaxis(background_image, 0, -1)
                background_image = Image.fromarray(background_image.astype('uint8'))
            pil_frame = background_image.resize((width, height)).convert("RGBA")
        else:
            background_color = GRCounterVideo._available_colours.get(background_colour.lower(), "#000000")
            background_color = GRCounterVideo.hex_to_rgba(background_color, 100)
            pil_frame = Image.new('RGBA', (width, height), background_color)

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
        pil_frame = Image.alpha_composite(pil_frame, text_layer)
        frame = np.array(pil_frame)  # Convert back to RGBA

        return frame

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
    def generate_video(cls, start, finish, countdown, clock_type, output_path, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image=False, background_image=None, resolution='HD 1280x720', counter_duration=1000, rotate=False, rotate_type="clockwise", rotate_freq=1, processing_device="cpu", start_x=0, start_y=0, end_x=0, end_y=0, movement="constant", upscale=1.0):
        # Append date and time to the output path
        now = datetime.now().strftime("%d-%y-%m-%H-%M")
        output_path = os.path.splitext(output_path)[0] + f"_{now}.mp4"

        frames = cls.generate_frames(start, finish, countdown, clock_type, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, background_image, resolution, counter_duration, fps, rotate, rotate_type, rotate_freq, device=processing_device, start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y, movement=movement)
        output_file = save_video(frames, output_path, fps, background_colour)
        file_details = cls.get_file_details(output_file, start, finish, countdown, clock_type, fps, font_path, font_size_min, font_size_max, font_color, font_opacity, font_shadow, font_shadow_dist, font_control, outline, outline_size, outline_color, outline_opacity, background_colour, use_background_image, resolution, counter_duration, start_x, start_y, end_x, end_y, movement)
        
        # Ensure frames are in the correct format before converting to tensors
        for frame in frames:
            print(f"Frame shape before permutation: {frame.shape}")
        frame_images = [torch.from_numpy(frame).permute(0, 1, 2) for frame in frames]  # Convert to (C, H, W) format
        batch_tensor = torch.stack(frame_images)  # Create a batch in (N, C, H, W) format

        # Upscale the batch tensor using the upscale method if upscale >= 1.0
        if upscale >= 1.0:
            upscaled_batch_tensor = cls.upscale(batch_tensor, "lanczos", upscale)
        else:
            upscaled_batch_tensor = cls.upscale(batch_tensor, "lanczos", upscale)

        return (file_details, frame_images, upscaled_batch_tensor)
