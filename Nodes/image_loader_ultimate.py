"""
ComfyUI Custom Node - Image Selector with Interactive Cropper
Place this file in: ComfyUI/custom_nodes/image_loader_ultimate/image_loader_ultimate.py
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import folder_paths

class ImageLoaderUltimateNode:
    """
    A node that lets users select an image file and interactively crop and rotate it
    using draggable bars and rotate buttons in the UI.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Image path will appear here"
                }),
            },
            "hidden": {
                "crop_data": ("STRING", {"default": "{}"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_image"
    CATEGORY = "image"
    
    def crop_image(self, image_path, crop_data="{}"):
        """
        Load the selected image, apply rotation, then crop.
        image_path is encoded as "path|x,y,w,h,rotation" by the JS frontend.
        """
        print(f"\n=== Image Loader Ultimate ===")
        print(f"Image path raw: '{image_path}'")
        print(f"Crop data fallback: '{crop_data}'")

        # ── Parse embedded crop + rotation from image_path ─────────────────
        embedded_crop = None
        actual_path   = image_path
        rotation      = 0
        flip_h        = False
        flip_v        = False

        if image_path and "|" in image_path:
            parts      = image_path.split("|", 1)
            actual_path = parts[0]
            try:
                nums = [int(v) for v in parts[1].split(",")]
                embedded_crop = {"x": nums[0], "y": nums[1], "width": nums[2], "height": nums[3]}
                rotation = nums[4] if len(nums) > 4 else 0
                flip_h   = bool(nums[5]) if len(nums) > 5 else False
                flip_v   = bool(nums[6]) if len(nums) > 6 else False
                print(f"Embedded crop: {embedded_crop}, rotation: {rotation}°, flipH: {flip_h}, flipV: {flip_v}")
            except Exception as ex:
                print(f"Could not parse embedded crop: {ex}")

        # ── Locate the image file ───────────────────────────────────────────
        input_dir = folder_paths.get_input_directory()
        full_image_path = None

        if actual_path:
            for candidate in [
                os.path.join(input_dir, actual_path),
                actual_path,
                os.path.join(input_dir, "cropper_uploads", actual_path),
                os.path.join(input_dir, os.path.basename(actual_path)),
            ]:
                if os.path.exists(candidate):
                    full_image_path = candidate
                    print(f"Found image at: {full_image_path}")
                    break

        if not full_image_path:
            print(f"❌ No image found: {actual_path}")
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            mask  = torch.zeros((1, 512, 512),    dtype=torch.float32)
            return (blank, mask)

        try:
            # ── Resolve crop region ─────────────────────────────────────────
            # Prefer embedded (always serialized); fall back to crop_data widget
            if embedded_crop is not None:
                crop_src = embedded_crop
            else:
                try:
                    crop_src = json.loads(crop_data)
                    rotation = crop_src.get("rotation", 0)
                    flip_h   = bool(crop_src.get("flipH", False))
                    flip_v   = bool(crop_src.get("flipV", False))
                except Exception:
                    crop_src = {}

            crop_x = max(0, crop_src.get("x", 0))
            crop_y = max(0, crop_src.get("y", 0))
            crop_w = max(1, crop_src.get("width", 100))
            crop_h = max(1, crop_src.get("height", 100))
            print(f"Crop - x:{crop_x} y:{crop_y} w:{crop_w} h:{crop_h}  rotation:{rotation}°")

            # ── Load, rotate, flip ──────────────────────────────────────────
            img = Image.open(full_image_path).convert("RGB")
            print(f"Loaded: {img.size[0]}×{img.size[1]}")

            # PIL rotates counter-clockwise, so negate for CW convention
            if rotation == 90:
                img = img.rotate(-90, expand=True)   # 90° CW
            elif rotation == 180:
                img = img.rotate(180, expand=True)
            elif rotation == 270:
                img = img.rotate(90, expand=True)    # 270° CW == 90° CCW

            if flip_h:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_v:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            rot_w, rot_h = img.size
            print(f"After rotation/flip: {rot_w}×{rot_h}")

            # ── Clamp crop to rotated image bounds ──────────────────────────
            crop_x = max(0, min(crop_x, rot_w - 1))
            crop_y = max(0, min(crop_y, rot_h - 1))
            crop_w = max(1, min(crop_w, rot_w - crop_x))
            crop_h = max(1, min(crop_h, rot_h - crop_y))
            print(f"Clamped crop - x:{crop_x} y:{crop_y} w:{crop_w} h:{crop_h}")

            # ── Crop ────────────────────────────────────────────────────────
            cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

            # ── Convert to ComfyUI tensors ──────────────────────────────────
            arr        = np.array(cropped).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(arr).unsqueeze(0)          # (1, H, W, 3)

            # Mask: full rotated-image size, 1s only inside the crop region.
            # ComfyUI convention: (1, H, W), 1 = selected/visible area.
            mask = torch.zeros((1, rot_h, rot_w), dtype=torch.float32)
            mask[0, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = 1.0

            print(f"✓ Image shape: {img_tensor.shape}  Mask shape: {mask.shape}")
            return (img_tensor, mask)

        except Exception as ex:
            print(f"❌ Error: {ex}")
            import traceback; traceback.print_exc()
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            mask  = torch.zeros((1, 512, 512),    dtype=torch.float32)
            return (blank, mask)


NODE_CLASS_MAPPINGS = {
    "ImageLoaderUltimate": ImageLoaderUltimateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoaderUltimate": "Image Loader Ultimate 🖼️"
}