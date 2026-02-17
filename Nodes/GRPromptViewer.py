import os
from datetime import datetime
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

class GRPromptViewer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moondream_model = None
        self.moondream_tokenizer = None
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-evaluate to refresh file list
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        # Get the directory where this file is located (custom node directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Look for prompts folder in the custom node directory
        prompts_dir = os.path.join(current_dir, "prompts")

        # Get list of folders in prompts directory
        folders = ["(root)"]  # Default option for files in root prompts folder
        all_files = set()  # Collect all files from all folders

        if os.path.exists(prompts_dir):
            # Get folders
            for item in os.listdir(prompts_dir):
                item_path = os.path.join(prompts_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)

                    # Get text files in this folder
                    text_files = []
                    image_files = []
                    
                    for f in os.listdir(item_path):
                        if os.path.isfile(os.path.join(item_path, f)):
                            if f.endswith(('.txt', '.log', '.json', '.csv', '.md')):
                                text_files.append(f)
                                all_files.add(f)
                            elif f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                image_files.append(f)
                    
                    # Add images that don't have corresponding text files
                    for img in image_files:
                        base_name = os.path.splitext(img)[0]
                        has_txt = f"{base_name}.txt" in text_files
                        if not has_txt:
                            all_files.add(img)

            # Also get files in root
            text_files = []
            image_files = []
            
            for f in os.listdir(prompts_dir):
                if os.path.isfile(os.path.join(prompts_dir, f)):
                    if f.endswith(('.txt', '.log', '.json', '.csv', '.md')):
                        text_files.append(f)
                        all_files.add(f)
                    elif f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        image_files.append(f)
            
            # Add images that don't have corresponding text files
            for img in image_files:
                base_name = os.path.splitext(img)[0]
                has_txt = f"{base_name}.txt" in text_files
                if not has_txt:
                    all_files.add(img)

        # If no files found, add placeholder
        if not all_files:
            all_files = {"No files found"}
        else:
            # Add "No files found" to the list so it's always valid
            all_files.add("No files found")

        # Always include the Random sentinel so ComfyUI considers it a valid value
        all_files.add("\U0001f3b2 Random")

        # Sort remaining files, but Random will be inserted first by JS
        sorted_files = ["\U0001f3b2 Random"] + sorted(
            [f for f in all_files if f != "\U0001f3b2 Random"]
        )

        return {
            "required": {
                "folder": (sorted(folders),),
                "file": (sorted_files,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "content": ("STRING", {"multiline": True, "default": ""}),
                "edited": ("BOOLEAN", {"default": False}),
            },
            "optional": {},
        }

    # ADD IMAGE OUTPUT TYPE
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("content", "image")
    FUNCTION = "read_file"
    CATEGORY = "Custom"
    OUTPUT_NODE = True

    RANDOM_SENTINEL = "\U0001f3b2 Random"

    def _get_files_in_folder(self, folder):
        """Return a sorted list of valid prompt files (text + orphan images) in the given folder."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, "prompts")

        if folder == "(root)":
            target_dir = prompts_dir
        else:
            target_dir = os.path.join(prompts_dir, folder)

        if not os.path.exists(target_dir):
            return []

        text_files = [f for f in os.listdir(target_dir)
                      if os.path.isfile(os.path.join(target_dir, f))
                      and f.endswith(('.txt', '.log', '.json', '.csv', '.md'))]

        image_files = [f for f in os.listdir(target_dir)
                       if os.path.isfile(os.path.join(target_dir, f))
                       and f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]

        files_set = set(text_files)
        for img in image_files:
            base_name = os.path.splitext(img)[0]
            if f"{base_name}.txt" not in text_files:
                files_set.add(img)

        return sorted(files_set)

    def read_file(self, folder, file, seed=0, content="", edited=False):
        """
        Read and return file content with optional image output.
        When file == RANDOM_SENTINEL the seed is used to deterministically
        pick a file from the folder on every execution.
        """

        print(f"=== GRPromptViewer.read_file called ===")
        print(f"folder: {folder}")
        print(f"file: {file}")
        print(f"seed: {seed}")
        print(f"content: {'None' if content is None else f'length={len(content)}'}")
        print(f"edited flag: {edited} (type: {type(edited)})")

        # --- RANDOM MODE: resolve actual file from seed ---
        random_mode_active = (file == self.RANDOM_SENTINEL)
        if random_mode_active:
            available = self._get_files_in_folder(folder)
            if available:
                # Use seed to pick deterministically; seed changes -> different file
                chosen = available[seed % len(available)]
                print(f"Random mode: seed={seed} -> picked '{chosen}' from {len(available)} files")
                file = chosen
            else:
                file = "No files found"
                print("Random mode: no files available in folder")

        # In random mode, always read the resolved file from disk so the
        # seed drives the output regardless of the cached content widget.
        if random_mode_active and file not in ("No files found", "Select folder first", ""):
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            _prm_dir = os.path.join(_cur_dir, "prompts")
            _rnd_path = os.path.join(_prm_dir, file) if folder == "(root)" else os.path.join(_prm_dir, folder, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                try:
                    if os.path.exists(_rnd_path) and os.path.abspath(_rnd_path).startswith(os.path.abspath(_prm_dir)):
                        with open(_rnd_path, 'r', encoding='utf-8') as _fh:
                            content = _fh.read()
                        print(f"Random mode: loaded content from disk ({len(content)} chars)")
                except Exception as _e:
                    print(f"Random mode: failed to read file from disk: {_e}")

        # Convert edited to boolean
        edited_bool = edited in [True, "true", "True", "1", 1]

        # Check if content exists (either from input or from preview)
        has_content = content is not None and content != ""
        print(f"has_content: {has_content}, edited_bool: {edited_bool}")

        # LOGIC:
        # 1. If we have content, use it (whether from input connection or preview)
        if has_content:
            final_content = content
            if edited_bool:
                print(f"Using content from edited preview, length: {len(content)}")
            else:
                print(f"Using content (from input or loaded file), length: {len(content)}")
        else:
            # No content available
            final_content = "No content available"
            print(f"No content available")

        # AUTO-SAVE LOGIC
        # Auto-save when content is meaningful and edited
        has_meaningful_content = final_content and final_content.strip() and len(final_content) > 10

        if has_meaningful_content and edited_bool:
            self._auto_save(final_content)
            print(f"Auto-saved edited content")
        else:
            print(f"Auto-save skipped (edited: {edited_bool}, meaningful: {has_meaningful_content})")

        # TRY TO LOAD THE CURRENT IMAGE FOR OUTPUT
        image_output = None
        
        # Get current directory and prompts directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, "prompts")
        
        # Build file path based on folder
        if folder == "(root)":
            file_path = os.path.join(prompts_dir, file)
        else:
            file_path = os.path.join(prompts_dir, folder, file)
        
        # Check if this is an image file
        if file and file != "No files found" and file != "Select folder first":
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                # This is an image file - try to load it
                try:
                    if os.path.exists(file_path) and os.path.abspath(file_path).startswith(os.path.abspath(prompts_dir)):
                        image = Image.open(file_path).convert("RGB")
                        
                        # Convert to tensor in the format ComfyUI expects
                        image_np = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np)[None, ...]  # Add batch dimension
                        
                        image_output = image_tensor
                        print(f"✓ Loaded image: {file}, shape: {image_tensor.shape}")
                    else:
                        print(f"✗ Image file not found or invalid path: {file_path}")
                        # Create empty image tensor
                        image_output = torch.zeros((1, 64, 64, 3))
                except Exception as e:
                    print(f"✗ Error loading image {file}: {e}")
                    # Create empty image tensor
                    image_output = torch.zeros((1, 64, 64, 3))
            else:
                # This is a text file - check if there's a corresponding image
                base_name = os.path.splitext(file)[0]
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
                
                for ext in image_extensions:
                    image_file = base_name + ext
                    if folder == "(root)":
                        image_path = os.path.join(prompts_dir, image_file)
                    else:
                        image_path = os.path.join(prompts_dir, folder, image_file)
                    
                    if os.path.exists(image_path) and os.path.abspath(image_path).startswith(os.path.abspath(prompts_dir)):
                        try:
                            image = Image.open(image_path).convert("RGB")
                            
                            # Convert to tensor in the format ComfyUI expects
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np)[None, ...]  # Add batch dimension
                            
                            image_output = image_tensor
                            print(f"✓ Loaded associated image: {image_file}, shape: {image_tensor.shape}")
                            break
                        except Exception as e:
                            print(f"✗ Error loading associated image {image_file}: {e}")
                            continue
                
                if image_output is None:
                    # No image found, create empty tensor
                    image_output = torch.zeros((1, 64, 64, 3))
                    print(f"✗ No associated image found for text file: {file}")
        else:
            # No valid file selected, create empty tensor
            image_output = torch.zeros((1, 64, 64, 3))
            print(f"✗ No file selected or invalid file name")

        print(f"Final content length: {len(final_content)}")
        print(f"Image output type: {type(image_output)}, shape: {image_output.shape if hasattr(image_output, 'shape') else 'No shape'}")

        # Return both content and image
        return {
            "ui": {"text": [final_content]},
            "result": (final_content, image_output),
        }

    def _auto_save(self, content):
        """
        Auto-save the content to auto-save folder with timestamp filename
        """
        try:
            # Get current directory and create auto-save folder path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(current_dir, "prompts")
            autosave_dir = os.path.join(prompts_dir, "auto-save")

            # Create auto-save directory if it doesn't exist
            if not os.path.exists(autosave_dir):
                os.makedirs(autosave_dir)

            # Generate timestamp: dd-mm-yyyy-hh-mm-ss
            timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

            # Get first 6 words and sanitize for filename
            words = content.split()[:6]
            first_words = " ".join(words)

            # Remove characters that aren't safe for filenames
            safe_words = "".join(c for c in first_words if c.isalnum() or c in (' ', '-', '_'))
            safe_words = safe_words.strip()

            # Build filename: timestamp + first words + extension
            if safe_words:
                filename = f"{timestamp} {safe_words}.autosave.txt"
            else:
                filename = f"{timestamp}.autosave.txt"

            filepath = os.path.join(autosave_dir, filename)

            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Auto-saved to: {filename}")

        except Exception as e:
            print(f"Auto-save failed: {str(e)}")
    
    def load_moondream_model(self):
        """Load Moondream2 model (cached)"""
        if self.moondream_model is not None and self.moondream_tokenizer is not None:
            print("Using cached Moondream2 model")
            return self.moondream_tokenizer, self.moondream_model
        
        try:
            print(f"\nLoading Moondream2 model: vikhyatk/moondream2\n")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.moondream_tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2", 
                trust_remote_code=True
            )
            
            self.moondream_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            
            self.moondream_model.eval()
            print("Model loaded successfully!\n")
            return self.moondream_tokenizer, self.moondream_model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None, None

    def generate_caption_for_image(self, image_path, max_length=1024):
        """Generate caption using Moondream2"""
        try:
            tokenizer, model = self.load_moondream_model()
            
            if not tokenizer or not model:
                return "Error: Failed to load Moondream2 model"
            
            image = Image.open(image_path).convert("RGB")
            enc_image = model.encode_image(image)
            
            question = "Describe this image in detail."
            with torch.no_grad():
                answer = model.answer_question(enc_image, question, tokenizer)
            
            caption = answer.strip()
            
            # Clean up
            patterns_to_remove = [
                question,
                f"{question}:",
                f"Question: {question}",
                f"Q: {question}",
            ]
            
            for pattern in patterns_to_remove:
                if caption.lower().startswith(pattern.lower()):
                    caption = caption[len(pattern):].lstrip(" :-")
                    break
            
            if caption.lower().startswith("answer:"):
                caption = caption[7:].strip()
            elif caption.lower().startswith("a:"):
                caption = caption[2:].strip()
            
            # Truncate if needed
            if len(caption) > max_length:
                caption = caption[:max_length].rsplit(' ', 1)[0] + "..."
            
            return caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"Caption generation failed: {str(e)}"


# Node class mapping
NODE_CLASS_MAPPINGS = {
    "GRPromptViewer": GRPromptViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptViewer": "GR Prompt Viewer",
}