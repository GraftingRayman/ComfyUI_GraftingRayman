import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import random
import cv2
import numpy as np


class GRPanOrZoom:
    def __init__(self):
        self.depth_model = None
        self.initial_center = None  # For weighted-singular mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_maps": ("IMAGE",),
                "zoom": ("FLOAT", {"default": 1.5, "min": 1.1, "max": 5.0, "step": 0.1}),
                "frames_per_transition": ("INT", {"default": 24, "min": 1, "max": 1200, "step": 1}),
                "mode": (["pan-left", "pan-right", "pan-up", "pan-down", "zoom-in", "zoom-out"], {"default": "pan-left"}),
                "use_depth": ("BOOLEAN", {"default": False}),
                "depth_focus_method": (["centroid", "max-point", "adaptive-region", "weighted-average", "smoothed", "weighted-singular"], {"default": "weighted-average"}),
                "max_depth": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "parallax_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "randomize": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("poz_frames",)
    FUNCTION = "apply_pan_or_zoom"
    CATEGORY = "GraftingRayman/Video"

    def compute_depth_position(self, depth_map: torch.Tensor, method: str) -> Tuple[int, int]:
        """
        Compute the depth-based focus point (x, y) using the specified method.
        """
        depth_map_np = depth_map.squeeze().cpu().numpy()
        if depth_map_np.ndim > 2:
            depth_map_np = depth_map_np[0]

        h, w = depth_map_np.shape

        if method == "centroid":
            _, binary_map = cv2.threshold(depth_map_np, depth_map_np.max() * 0.9, 255, cv2.THRESH_BINARY)
            binary_map = binary_map.astype(np.uint8)
            moments = cv2.moments(binary_map)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = w // 2, h // 2

        elif method == "max-point":
            max_depth_val, max_depth_idx = torch.max(depth_map.reshape(-1), dim=0)
            cy = max_depth_idx // w
            cx = max_depth_idx % w

        elif method == "adaptive-region":
            _, binary_map = cv2.threshold(depth_map_np, depth_map_np.max() * 0.9, 255, cv2.THRESH_BINARY)
            binary_map = binary_map.astype(np.uint8)
            x, y, width, height = cv2.boundingRect(binary_map)
            cx, cy = x + width // 2, y + height // 2

        elif method in ["weighted-average", "weighted-singular"]:
            y_coords, x_coords = np.indices((h, w))
            total_depth = np.sum(depth_map_np)
            if total_depth != 0:
                cx = int(np.sum(x_coords * depth_map_np) / total_depth)
                cy = int(np.sum(y_coords * depth_map_np) / total_depth)
            else:
                cx, cy = w // 2, h // 2

            if method == "weighted-singular" and self.initial_center is None:
                self.initial_center = (cy, cx)
            elif method == "weighted-singular":
                cy, cx = self.initial_center

        elif method == "smoothed":
            smoothed_map = cv2.GaussianBlur(depth_map_np, (15, 15), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(smoothed_map)
            cx, cy = max_loc

        else:
            cx, cy = w // 2, h // 2

        return cy, cx

    def process_frame(self, i: int, image: torch.Tensor, depth_map: torch.Tensor, zoom: float, frames_per_transition: int, mode: str, use_depth: bool, depth_focus_method: str, max_depth: float, parallax_strength: float, device: str) -> torch.Tensor:
        """
        Process a single frame for pan/zoom effect.
        """
        current_image = image.to(device)
        h, w = current_image.shape[1:3]

        # Calculate current zoom level
        if mode == "zoom-in":
            current_zoom = 1 + (zoom - 1) * (i / frames_per_transition)
        elif mode == "zoom-out":
            current_zoom = zoom - (zoom - 1) * (i / frames_per_transition)
        else:
            current_zoom = zoom

        new_h, new_w = int(h * current_zoom), int(w * current_zoom)

        # Calculate crop coordinates
        if use_depth:
            # Get depth-based center
            pan_y, pan_x = self.compute_depth_position(depth_map, depth_focus_method)

            # Adjust pan_x and pan_y for zoomed dimensions
            pan_x = int(pan_x * (new_w / w))
            pan_y = int(pan_y * (new_h / h))

            # Calculate crop coordinates based on depth center
            pan_crop_x = max(0, min(new_w - w, pan_x - w // 2))
            pan_crop_y = max(0, min(new_h - h, pan_y - h // 2))
        else:
            # Default to center if depth is not used
            pan_crop_x = (new_w - w) // 2
            pan_crop_y = (new_h - h) // 2

        # Adjust for pan modes
        if mode in ["pan-left", "pan-right", "pan-up", "pan-down"]:
            pan_step_x = (new_w - w) // frames_per_transition
            pan_step_y = (new_h - h) // frames_per_transition

            if mode == "pan-left":
                pan_crop_x = new_w - w - i * pan_step_x
            elif mode == "pan-right":
                pan_crop_x = i * pan_step_x
            elif mode == "pan-up":
                pan_crop_y = new_h - h - i * pan_step_y
            elif mode == "pan-down":
                pan_crop_y = i * pan_step_y

        # Apply zoom and crop
        zoomed = F.interpolate(
            current_image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        frame = zoomed[:, pan_crop_y:pan_crop_y + h, pan_crop_x:pan_crop_x + w]
        return frame.cpu()

    def apply_pan_or_zoom(self, images: torch.Tensor, depth_maps: torch.Tensor, zoom: float, frames_per_transition: int, mode: str, use_depth: bool, depth_focus_method: str, max_depth: float, parallax_strength: float, device: str, randomize: bool) -> Tuple[torch.Tensor]:
        """
        Apply pan/zoom effect to a batch of images.
        """
        if len(images.shape) == 4:
            images = images.permute(0, 3, 1, 2).to(device)
            depth_maps = depth_maps.permute(0, 3, 1, 2).to(device)

        frames = []
        modes = ["pan-left", "pan-right", "pan-up", "pan-down", "zoom-in", "zoom-out"]
        self.initial_center = None

        with tqdm(total=frames_per_transition * len(images), desc="Generating frames", unit="frame") as pbar:
            for image, depth_map in zip(images, depth_maps):
                current_mode = random.choice(modes) if randomize else mode
                for i in range(frames_per_transition):
                    frame = self.process_frame(i, image, depth_map, zoom, frames_per_transition, current_mode, use_depth, depth_focus_method, max_depth, parallax_strength, device)
                    frames.append(frame)
                    pbar.update(1)

        output = torch.stack(frames, dim=0)
        output = output.permute(0, 2, 3, 1)
        return (output,)


# Export the class
__all__ = ['GRPanOrZoom']
