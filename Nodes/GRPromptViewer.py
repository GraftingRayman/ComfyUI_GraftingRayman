import os
from datetime import datetime

class GRPromptViewer:
    
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
                    # Get files in this folder
                    for f in os.listdir(item_path):
                        if os.path.isfile(os.path.join(item_path, f)) and f.endswith(('.txt', '.log', '.json', '.csv', '.md')):
                            all_files.add(f)
            
            # Also get files in root
            for f in os.listdir(prompts_dir):
                if os.path.isfile(os.path.join(prompts_dir, f)) and f.endswith(('.txt', '.log', '.json', '.csv', '.md')):
                    all_files.add(f)
        
        # If no files found, add placeholder
        if not all_files:
            all_files = {"No files found"}
        else:
            # Add "No files found" to the list so it's always valid
            all_files.add("No files found")
        
        return {
            "required": {
                "folder": (sorted(folders),),
                "file": (sorted(list(all_files)),),  # List all files from all folders
                "content": ("STRING", {"multiline": True, "default": ""}),  # REMOVED forceInput - now accepts both widget and input
                "edited": ("BOOLEAN", {"default": False}),  # Moved to required so it's always passed
            },
            "optional": {
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "read_file"
    CATEGORY = "Custom"
    OUTPUT_NODE = True
    
    def read_file(self, folder, file, content="", edited=False):
        """
        Read and return file content
        Logic:
        1. If content input is connected (has content AND edited=False) → use connected content
        2. If content exists and edited=True → use edited content from preview
        3. Otherwise → return empty/no content message
        
        Note: File reading from disk only happens in JS when file selection changes
        """
        print(f"=== GRPromptViewer.read_file called ===")
        print(f"folder: {folder}")
        print(f"file: {file}")
        print(f"content: {'None' if content is None else f'length={len(content)}'}")
        print(f"edited flag: {edited} (type: {type(edited)})")
        
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
        
        print(f"Final content length: {len(final_content)}")
        
        # Return both as output AND send to UI for display
        return {"ui": {"text": [final_content]}, "result": (final_content,)}
    
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
