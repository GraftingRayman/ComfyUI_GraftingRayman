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
            },
            "optional": {
                "content": ("STRING", {"multiline": True, "default": "", "forceInput": True}),  # Accept from other nodes
            },
            "hidden": {
                "edited": ("BOOLEAN", {"default": False}),  # Hidden input to track if content was edited
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "read_file"
    CATEGORY = "Custom"
    OUTPUT_NODE = True
    
    def read_file(self, folder, file, content=None, edited=False):
        """
        Read and return file content from the prompts directory
        Always prefer the content parameter when it has meaningful data
        """
        print(f"=== GRPromptViewer.read_file called ===")
        print(f"folder: {folder}")
        print(f"file: {file}")
        print(f"content: {content if content is None else f'length={len(content)}'}")
        print(f"edited flag: {edited}")
        print(f"edited flag type: {type(edited)}")
        
        # Check if content came from an input connection (optional parameter)
        content_from_input = content is not None and content != ""
        
        # Define what constitutes "empty" or placeholder content
        placeholder_texts = [
            "Select a folder and file to view contents...",
            "Please select a folder and file", 
            "No file selected",
            "No files found in this folder",
            "Loading...",
            "Error loading file:"
        ]
        
        content = content or ""  # Convert None to empty string
        is_placeholder = any(placeholder in content for placeholder in placeholder_texts) if content else True
        has_meaningful_content = content and content.strip() and not is_placeholder
        
        # DEBUG: Log decision process
        print(f"content_from_input: {content_from_input}")
        print(f"has_meaningful_content: {has_meaningful_content}")
        print(f"is_placeholder: {is_placeholder}")
        
        # Always prefer the content parameter when it has meaningful data
        if has_meaningful_content:
            final_content = content
            print(f"Using content from parameter (length: {len(content)}, meaningful: {has_meaningful_content})")
        elif file == "Select folder first" or file == "No files found" or is_placeholder:
            final_content = content  # Use whatever content we have
            print(f"Using placeholder content")
        else:
            # Fall back to reading from file only if we don't have meaningful content
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(current_dir, "prompts")
            
            if folder == "(root)":
                file_path = os.path.join(prompts_dir, file)
            else:
                file_path = os.path.join(prompts_dir, folder, file)
            
            if not os.path.exists(file_path):
                final_content = f"File not found: {file}"
                print(f"File not found: {file_path}")
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        final_content = f.read()
                    print(f"Loaded file from disk: {file}, length: {len(final_content)}")
                except Exception as e:
                    final_content = f"Error reading file: {str(e)}"
                    print(f"Error reading file: {str(e)}")
        
        # AUTO-SAVE LOGIC: Save if content was edited OR if content came from input connection
        # Convert edited to boolean to handle string "true"/"false" from JS
        edited_bool = edited in [True, "true", "True", "1", 1]
        
        # Auto-save when:
        # 1. User manually edited (edited flag is true)
        # 2. Content came from an input connection (another node)
        should_auto_save = (edited_bool or content_from_input) and has_meaningful_content
        
        if should_auto_save:
            self._auto_save(final_content)
            if content_from_input and not edited_bool:
                print(f"Content from input connection (another node), auto-saved")
            else:
                print(f"Content was edited by user, auto-saved")
        else:
            print(f"Auto-save skipped - edited: {edited_bool}, from_input: {content_from_input}, meaningful: {has_meaningful_content}")
        
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
            # Keep only alphanumeric, spaces, hyphens, and underscores
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
