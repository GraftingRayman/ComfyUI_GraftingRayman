"""
ComfyUI Custom Node - Image Loader Ultimate Multi
Place in: ComfyUI/custom_nodes/image_loader_ultimate_multi/image_loader_ultimate_multi.py

Returns:
    image1      - cropped/rotated/flipped Image 1
    image2      - cropped/rotated/flipped Image 2
    background  - cropped/rotated/flipped Background
    composed    - background with image1 and image2 composited at their positions
    mask1       - full-bg-size mask with 1s where image1 is placed
    mask2       - full-bg-size mask with 1s where image2 is placed
"""

import os
import torch
import numpy as np
from PIL import Image
import folder_paths


def _parse_encoded(raw):
    """Parse 'path|cropX,cropY,cropW,cropH,rotation,flipH,flipV,posX,posY'"""
    if not raw:
        return None
    file_path = raw
    crop_x=0; crop_y=0; crop_w=None; crop_h=None
    rotation=0; flip_h=False; flip_v=False; pos_x=0; pos_y=0
    if "|" in raw:
        path_part, num_part = raw.split("|", 1)
        file_path = path_part
        try:
            nums = [int(v) for v in num_part.split(",")]
            crop_x   = nums[0] if len(nums) > 0 else 0
            crop_y   = nums[1] if len(nums) > 1 else 0
            crop_w   = nums[2] if len(nums) > 2 else None
            crop_h   = nums[3] if len(nums) > 3 else None
            rotation = nums[4] if len(nums) > 4 else 0
            flip_h   = bool(nums[5]) if len(nums) > 5 else False
            flip_v   = bool(nums[6]) if len(nums) > 6 else False
            pos_x    = nums[7] if len(nums) > 7 else 0
            pos_y    = nums[8] if len(nums) > 8 else 0
        except Exception as ex:
            print(f"[ILU-Multi] parse error: {ex}")
    return dict(file_path=file_path, crop_x=crop_x, crop_y=crop_y,
                crop_w=crop_w, crop_h=crop_h, rotation=rotation,
                flip_h=flip_h, flip_v=flip_v, pos_x=pos_x, pos_y=pos_y)


def _find_file(file_path):
    input_dir = folder_paths.get_input_directory()
    for candidate in [
        os.path.join(input_dir, file_path),
        file_path,
        os.path.join(input_dir, "ilu_multi_uploads", file_path),
        os.path.join(input_dir, os.path.basename(file_path)),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_and_transform(p):
    """Load image, rotate, flip, then crop. Returns RGB PIL image or None."""
    if not p or not p.get("file_path"):
        return None
    full_path = _find_file(p["file_path"])
    if not full_path:
        print(f"[ILU-Multi] ❌ Not found: {p['file_path']}")
        return None

    img = Image.open(full_path).convert("RGB")

    rot = p["rotation"]
    if rot == 90:    img = img.rotate(-90, expand=True)
    elif rot == 180: img = img.rotate(180,  expand=True)
    elif rot == 270: img = img.rotate(90,   expand=True)
    if p["flip_h"]:  img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if p["flip_v"]:  img = img.transpose(Image.FLIP_TOP_BOTTOM)

    w, h = img.size
    cx = max(0, min(p["crop_x"], w-1))
    cy = max(0, min(p["crop_y"], h-1))
    cw = max(1, min(p["crop_w"] or w, w-cx))
    ch = max(1, min(p["crop_h"] or h, h-cy))
    return img.crop((cx, cy, cx+cw, cy+ch))


def _remove_bg(img_rgb):
    """
    Remove background from an RGB PIL image using rembg.
    Returns RGBA PIL image with background made transparent.
    Falls back gracefully if rembg is not installed.
    """
    try:
        from rembg import remove as rembg_remove
        # rembg expects and returns bytes or PIL images
        rgba = rembg_remove(img_rgb)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        print(f"[ILU-Multi] Background removed ({img_rgb.size[0]}×{img_rgb.size[1]})")
        return rgba
    except ImportError:
        print("[ILU-Multi] ⚠️  rembg not installed — run: pip install rembg")
        print("[ILU-Multi]    Compositing without background removal.")
        return img_rgb.convert("RGBA")
    except Exception as ex:
        print(f"[ILU-Multi] ⚠️  rembg failed: {ex} — compositing without removal.")
        return img_rgb.convert("RGBA")


def _to_tensor(pil_img):
    # Ensure RGB before converting to tensor
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)   # (1, H, W, 3)


def _blank(h=512, w=512):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def _blank_mask(h=512, w=512):
    return torch.zeros((1, h, w), dtype=torch.float32)


class ImageLoaderUltimateMultiNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img1_data":         ("STRING",  {"default": "", "multiline": False}),
                "img2_data":         ("STRING",  {"default": "", "multiline": False}),
                "bg_data":           ("STRING",  {"default": "", "multiline": False}),
                "remove_background": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK",  "MASK")
    RETURN_NAMES  = ("image1", "image2", "background", "composed", "mask1", "mask2")
    FUNCTION      = "process"
    CATEGORY      = "image"

    def process(self, img1_data, img2_data, bg_data, remove_background=False):
        print(f"\n=== Image Loader Ultimate Multi (remove_bg={remove_background}) ===")

        p1  = _parse_encoded(img1_data)
        p2  = _parse_encoded(img2_data)
        pBG = _parse_encoded(bg_data)

        crop1 = _load_and_transform(p1)
        crop2 = _load_and_transform(p2)
        bg    = _load_and_transform(pBG)

        t1  = _to_tensor(crop1) if crop1 else _blank()
        t2  = _to_tensor(crop2) if crop2 else _blank()
        tBG = _to_tensor(bg)    if bg    else _blank()

        # ── Compose ──────────────────────────────────────────────────────────
        if bg is not None:
            bg_w, bg_h = bg.size
            composed = bg.copy().convert("RGBA")  # work in RGBA throughout

            mask1 = _blank_mask(bg_h, bg_w)
            mask2 = _blank_mask(bg_h, bg_w)

            # Resize: landscape bg → match height; portrait/square → match width
            bg_is_landscape = bg_w > bg_h

            def resize_to_bg(crop_img):
                if crop_img is None:
                    return None
                cw, ch = crop_img.size
                scale = (bg_h / ch) if bg_is_landscape else ((bg_h / 2) / ch)
                new_w = max(1, round(cw * scale))
                new_h = max(1, round(ch * scale))
                print(f"[ILU-Multi] Resize {cw}×{ch} → {new_w}×{new_h} Lanczos "
                      f"({'landscape' if bg_is_landscape else 'portrait'} bg {bg_w}×{bg_h})")
                return crop_img.resize((new_w, new_h), Image.LANCZOS)

            crop1_r = resize_to_bg(crop1)
            crop2_r = resize_to_bg(crop2)

            # Optionally strip backgrounds (returns RGBA)
            if remove_background:
                if crop1_r is not None:
                    crop1_r = _remove_bg(crop1_r)
                if crop2_r is not None:
                    crop2_r = _remove_bg(crop2_r)
            else:
                if crop1_r is not None:
                    crop1_r = crop1_r.convert("RGBA")
                if crop2_r is not None:
                    crop2_r = crop2_r.convert("RGBA")

            for idx, (crop, p, mask) in enumerate([(crop1_r, p1, mask1), (crop2_r, p2, mask2)]):
                if crop is None or p is None:
                    continue
                px = max(0, min(p["pos_x"], bg_w - 1))
                py = max(0, min(p["pos_y"], bg_h - 1))
                cw, ch = crop.size
                paste_w = min(cw, bg_w - px)
                paste_h = min(ch, bg_h - py)
                if paste_w <= 0 or paste_h <= 0:
                    continue

                paste_crop = crop.crop((0, 0, paste_w, paste_h))

                # Use alpha channel as paste mask so transparent pixels are skipped
                alpha_mask = paste_crop.split()[3]   # RGBA → A channel
                composed.paste(paste_crop, (px, py), mask=alpha_mask)

                # Build output mask from alpha (normalised 0–1)
                alpha_arr = np.array(alpha_mask).astype(np.float32) / 255.0
                mask[0, py:py+paste_h, px:px+paste_w] = torch.from_numpy(alpha_arr)
                print(f"[ILU-Multi] Image {idx+1} pasted at ({px},{py}) size {paste_w}×{paste_h}")

            tComposed = _to_tensor(composed)   # _to_tensor converts to RGB
        else:
            tComposed = _blank()
            mask1 = _blank_mask()
            mask2 = _blank_mask()

        print("[ILU-Multi] ✓ Done")
        return (t1, t2, tBG, tComposed, mask1, mask2)


NODE_CLASS_MAPPINGS = {
    "ImageLoaderUltimateMulti": ImageLoaderUltimateMultiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoaderUltimateMulti": "Image Loader Ultimate Multi 🖼️🖼️"
}