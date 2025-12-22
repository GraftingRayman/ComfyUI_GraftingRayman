import os
from comfy.sd import CLIP
import comfy.utils
import folder_paths
import nodes

class GRFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "custom": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("full_path", "last_folder", "filename_with_ext", "filename_no_ext", "custom")
    FUNCTION = "parse_path"
    CATEGORY = "utils"

    def parse_path(self, file_path, custom):
        # Normalize the path to handle both forward and backward slashes
        normalized_path = os.path.normpath(file_path)
        
        # Check if the path ends with a separator (indicates it's a directory path)
        is_directory = normalized_path.endswith(os.sep) or os.path.isdir(normalized_path)
        
        if is_directory or os.path.basename(normalized_path) == '':
            # Directory path case - no filename
            full_path = normalized_path.rstrip(os.sep)
            last_folder = os.path.basename(full_path) if full_path else ""
            
            # Return empty strings for filename outputs
            return (full_path, last_folder, "", "", custom)
        else:
            # File path case
            full_path = os.path.dirname(normalized_path)
            last_folder = os.path.basename(full_path) if full_path else ""
            filename_with_ext = os.path.basename(normalized_path)
            filename_no_ext = os.path.splitext(filename_with_ext)[0]
            
            return (full_path, last_folder, filename_with_ext, filename_no_ext, custom)