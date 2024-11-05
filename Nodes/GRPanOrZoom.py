import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from PIL import Image
from custom_controlnet_aux.depth_anything_v2.dpt import DepthAnythingV2
import random
from comfy.utils import ProgressBar  # Importing the ProgressBar

class GRPanOrZoom:
    def __init__(self):
        self.depth_model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "zoom": ("FLOAT", {"default": 3.0, "min": 1.1, "max": 5.0, "step": 0.1}),
                "frames_per_transition": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "mode": (["pan-left", "pan-right", "pan-up", "pan-down", "zoom-in", "zoom-out", "zoom-left", "zoom-right", "zoom-up", "zoom-down"], {"default": "pan-left"}),
                "use_depth": ("BOOLEAN", {"default": False}),
                "max_depth": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "randomize": ("BOOLEAN", {"default": False}),
                "zoom_through": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("poz_frames", "depth_maps")
    FUNCTION = "apply_zoom"
    CATEGORY = "Image Processing"

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor - tensor.min()  # Shift to non-negative
        tensor = tensor / (tensor.max() + 1e-5)  # Normalize to [0, 1]

        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='L')  # 'L' mode for grayscale
        return image.convert("RGB")  # Convert to RGB

    def compute_depth_zoom_position(self, depth_map: torch.Tensor, side: str = None) -> Tuple[int, int]:
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)

        h, w = depth_map.shape[-2:]
        
        if side in ["left", "right", "top", "bottom"]:
            if side == "left":
                target_region = depth_map[:, :, :, : w // 2]
            elif side == "right":
                target_region = depth_map[:, :, :, w // 2:]
            elif side == "top":
                target_region = depth_map[:, :, : h // 2, :]
            elif side == "bottom":
                target_region = depth_map[:, :, h // 2 :, :]
        else:
            target_region = depth_map

        max_depth_val, max_depth_idx = torch.max(target_region.reshape(-1), dim=0)
        max_depth_y = max_depth_idx // target_region.shape[-1]
        max_depth_x = max_depth_idx % target_region.shape[-1]

        if side == "right":
            max_depth_x += w // 2
        elif side == "bottom":
            max_depth_y += h // 2

        return int(max_depth_y), int(max_depth_x)

    def process_frame(self, i: int, image: torch.Tensor, zoom: float, frames_per_transition: int, mode: str, use_depth: bool, max_depth: float, device: str) -> torch.Tensor:
        current_image = image.to(device)
        h, w = current_image.shape[1:3]

        # Default zoom factor
        zoom_factor = 1.0  # Base zoom factor
        if mode in ["zoom-in", "zoom-out"]:
            if mode == "zoom-out":
                zoom_factor = 1.0 + (1.0 - 1.0 / zoom) * (i / frames_per_transition)
            elif mode == "zoom-in":
                zoom_factor = 1.0 + (zoom - 1.0) * (i / frames_per_transition)

        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # Set the zoom positions for new zoom modes
        zoom_y, zoom_x = h // 2, w // 2  # Default center

        if mode in ["zoom-left", "zoom-right", "zoom-up", "zoom-down"]:
            if mode == "zoom-left":
                zoom_x = w - 1  # Start from the far right
            elif mode == "zoom-right":
                zoom_x = 0  # Start from the far left
            elif mode == "zoom-up":
                zoom_y = h - 1  # Start from the bottom
            elif mode == "zoom-down":
                zoom_y = 0  # Start from the top
        
        # Perform the zoom
        zoomed = F.interpolate(
            current_image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        # Calculate step sizes for panning based on frames_per_transition
        pan_step_x = (new_w - w) // frames_per_transition
        pan_step_y = (new_h - h) // frames_per_transition

        # Adjust crop positions based on the mode and frame index
        if mode in ["pan-left", "pan-right", "pan-up", "pan-down"]:
            if mode == "pan-left":
                crop_x = new_w - w - i * pan_step_x
                crop_y = zoom_y - h // 2
            elif mode == "pan-right":
                crop_x = i * pan_step_x
                crop_y = zoom_y - h // 2
            elif mode == "pan-up":
                crop_y = new_h - h - i * pan_step_y
                crop_x = zoom_x - w // 2
            elif mode == "pan-down":
                crop_y = i * pan_step_y
                crop_x = zoom_x - w // 2
        elif mode in ["zoom-left", "zoom-right", "zoom-up", "zoom-down"]:
            crop_x = max(0, min(new_w - w, zoom_x - w // 2))
            crop_y = max(0, min(new_h - h, zoom_y - h // 2))

        # Crop the frame
        frame = zoomed[:, crop_y:crop_y + h, crop_x:crop_x + w]
        return frame.cpu()

    def apply_zoom(self, images: torch.Tensor, zoom: float, frames_per_transition: int, mode: str, use_depth: bool, max_depth: float, device: str, randomize: bool, zoom_through: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.permute(0, 3, 1, 2).to(device)

        frames = []
        depth_maps = []
        pan_modes = ["pan-left", "pan-right", "pan-up", "pan-down"]
        zoom_modes = ["zoom-in", "zoom-out", "zoom-left", "zoom-right", "zoom-up", "zoom-down"]
        modes = pan_modes + zoom_modes

        # Initialize ProgressBar with the total number of frames to process
        batch = len(images) * frames_per_transition
        pbar = ProgressBar(batch)

        for idx, image in enumerate(images):
            # Determine if we need to randomize based on selected mode
            current_mode = mode
            if randomize:
                if mode in pan_modes:
                    current_mode = random.choice(pan_modes)
                elif mode in zoom_modes:
                    current_mode = random.choice(zoom_modes)  # Randomize zoom modes

            # Compute depth map if `use_depth` is enabled
            if use_depth:
                if self.depth_model is None:
                    self.depth_model = DepthAnythingV2().to(device)
                depth_map = self.depth_model(image.unsqueeze(0), max_depth=max_depth)

                # Append the depth map tensor to the list
                depth_maps.append(depth_map.squeeze(0))  # Keep it as a tensor for return

            for i in range(frames_per_transition):
                # If zoom_through is enabled, we need to handle frame blending
                if zoom_through and idx < len(images) - 1 and i >= frames_per_transition - 1:
                    next_image = images[idx + 1]
                    # Blend the current frame with the next image's frame
                    blend_frame = self.process_frame(i - (frames_per_transition - 1), next_image, zoom, frames_per_transition, current_mode, use_depth, max_depth, device)
                    frame = self.process_frame(i, image, zoom, frames_per_transition, current_mode, use_depth, max_depth, device) * (1 - (i - (frames_per_transition - 1)) / (frames_per_transition - 1)) + blend_frame * ((i - (frames_per_transition - 1)) / (frames_per_transition - 1))
                else:
                    frame = self.process_frame(i, image, zoom, frames_per_transition, current_mode, use_depth, max_depth, device)

                frames.append(frame)
                pbar.update_absolute(idx * frames_per_transition + i)

        output = torch.stack(frames, dim=0)
        output = output.permute(0, 2, 3, 1)

        depth_output = torch.stack(depth_maps, dim=0) if depth_maps else torch.empty(0, device=device)

        return (output, self.tensor_to_image(depth_output))

# Export the class
__all__ = ['GRPanOrZoom']
