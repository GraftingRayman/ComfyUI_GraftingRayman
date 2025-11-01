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
        
        return {
            "required": {
                "folder": (sorted(folders),),
                "file": (sorted(list(all_files)),),  # List all files from all folders
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "read_file"
    CATEGORY = "Custom"
    OUTPUT_NODE = True
    
    def read_file(self, folder, file):
        """
        Read and return file content from the prompts directory
        """
        if file == "Select folder first" or file == "No files found":
            content = "Please select a folder and file"
        else:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(current_dir, "prompts")
            
            # Build file path based on folder selection
            if folder == "(root)":
                file_path = os.path.join(prompts_dir, file)
            else:
                file_path = os.path.join(prompts_dir, folder, file)
            
            if not os.path.exists(file_path):
                content = f"File not found: {file}"
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading file: {str(e)}"
        
        # Auto-save the content
        self._auto_save(content)
        
        # Return both as output AND send to UI for display
        return {"ui": {"text": [content]}, "result": (content,)}
    
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
            
            # Generate filename with timestamp: dd-mm-yy-hh-mm-ss.autosave.txt
            timestamp = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
            filename = f"{timestamp}.autosave.txt"
            filepath = os.path.join(autosave_dir, filename)
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Auto-saved to: {filename}")
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")
