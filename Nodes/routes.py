import os
import json
import server
from aiohttp import web
import folder_paths

# Global node instance for Moondream model
_gr_prompt_viewer_instance = None

def get_gr_prompt_viewer_instance():
    """Get or create the GRPromptViewer node instance for caption generation"""
    global _gr_prompt_viewer_instance
    if _gr_prompt_viewer_instance is None:
        try:
            from .GRPromptViewer import GRPromptViewer
            _gr_prompt_viewer_instance = GRPromptViewer()
            print("✓ GRPromptViewer instance created for caption generation")
        except Exception as e:
            print(f"Error creating GRPromptViewer instance: {e}")
            return None
    return _gr_prompt_viewer_instance


@server.PromptServer.instance.routes.get("/prompt_viewer/list_files")
async def list_files(request):
    """
    API endpoint to list files in a specific folder
    Lists both text files AND images (for caption generation workflow)
    """
    folder = request.query.get("folder", "(root)")
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Determine the target directory
    if folder == "(root)":
        target_dir = prompts_dir
    else:
        target_dir = os.path.join(prompts_dir, folder)
    
    # Security check
    if not os.path.abspath(target_dir).startswith(os.path.abspath(prompts_dir)):
        return web.json_response({"files": ["Invalid folder"]}, status=403)
    
    if not os.path.exists(target_dir):
        return web.json_response({"files": ["Folder not found"]}, status=404)
    
    try:
        # Get list of text files
        text_files = [f for f in os.listdir(target_dir) 
                     if os.path.isfile(os.path.join(target_dir, f)) 
                     and f.endswith(('.txt', '.log', '.json', '.csv', '.md'))]
        
        # Get list of image files
        image_files = [f for f in os.listdir(target_dir) 
                      if os.path.isfile(os.path.join(target_dir, f)) 
                      and f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
        
        # For each image, check if it has a corresponding text file
        # If not, add it to the list (for caption generation)
        files_to_show = set(text_files)
        
        for img in image_files:
            # Get base name without extension
            base_name = os.path.splitext(img)[0]
            
            # Check if there's a .txt file with the same base name
            has_txt = f"{base_name}.txt" in text_files
            
            # If no text file exists, add the image to the list
            if not has_txt:
                files_to_show.add(img)
        
        if not files_to_show:
            files = ["No files found"]
        else:
            files = sorted(list(files_to_show))
        
        return web.json_response({"files": files}, status=200)
    except Exception as e:
        return web.json_response({"files": [f"Error: {str(e)}"]}, status=500)


@server.PromptServer.instance.routes.get("/prompt_viewer/read")
async def read_prompt_file(request):
    """
    API endpoint to read files from the prompts directory
    """
    folder = request.query.get("folder", "(root)")
    filename = request.query.get("filename", "")
    
    if not filename:
        return web.Response(text="No filename provided", status=400)
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Build file path based on folder
    if folder == "(root)":
        file_path = os.path.join(prompts_dir, filename)
    else:
        file_path = os.path.join(prompts_dir, folder, filename)
    
    # Security check: ensure the file is within the prompts directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    if not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return web.Response(text=content, status=200)
    except Exception as e:
        return web.Response(text=f"Error reading file: {str(e)}", status=500)


@server.PromptServer.instance.routes.get("/prompt_viewer/read_image")
async def read_image_file(request):
    """
    API endpoint to read image files from the prompts directory
    """
    folder = request.query.get("folder", "(root)")
    filename = request.query.get("filename", "")
    
    if not filename:
        return web.Response(text="No filename provided", status=400)
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Build file path based on folder
    if folder == "(root)":
        file_path = os.path.join(prompts_dir, filename)
    else:
        file_path = os.path.join(prompts_dir, folder, filename)
    
    # Security check: ensure the file is within the prompts directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    if not os.path.exists(file_path):
        return web.Response(text="Image not found", status=404)
    
    # Check if it's an image file
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
        return web.Response(text="Not an image file", status=400)
    
    try:
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        # Determine content type based on file extension
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            content_type = 'image/jpeg'
        elif filename.lower().endswith('.gif'):
            content_type = 'image/gif'
        elif filename.lower().endswith('.bmp'):
            content_type = 'image/bmp'
        elif filename.lower().endswith('.webp'):
            content_type = 'image/webp'
        else:
            content_type = 'application/octet-stream'
        
        return web.Response(body=image_data, content_type=content_type)
    except Exception as e:
        return web.Response(text=f"Error reading image: {str(e)}", status=500)


@server.PromptServer.instance.routes.post("/prompt_viewer/save")
async def save_prompt_file(request):
    """
    API endpoint to save files to the prompts directory
    """
    try:
        data = await request.json()
    except:
        return web.Response(text="Invalid JSON", status=400)
    
    folder = data.get("folder", "(root)")
    filename = data.get("filename", "")
    content = data.get("content", "")
    
    if not filename:
        return web.Response(text="No filename provided", status=400)
    
    # Validate filename
    if not filename.endswith(('.txt', '.log', '.json', '.csv', '.md')):
        return web.Response(text="Invalid file extension", status=400)
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Track if we created a new folder
    folder_created = False
    
    # Build file path based on folder
    if folder == "(root)":
        file_path = os.path.join(prompts_dir, filename)
    else:
        folder_path = os.path.join(prompts_dir, folder)
        file_path = os.path.join(folder_path, filename)
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            folder_created = True
    
    # Security check: ensure the file is within the prompts directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return web.json_response({
            "message": "File saved successfully", 
            "filename": filename,
            "folder_created": folder_created
        }, status=200)
    except Exception as e:
        return web.Response(text=f"Error saving file: {str(e)}", status=500)


@server.PromptServer.instance.routes.get("/prompt_viewer/list_folders")
async def list_folders(request):
    """
    API endpoint to list all folders in the prompts directory
    """
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    folders = ["(root)"]  # Default option for files in root prompts folder
    
    if os.path.exists(prompts_dir):
        # Get folders
        for item in os.listdir(prompts_dir):
            item_path = os.path.join(prompts_dir, item)
            if os.path.isdir(item_path):
                folders.append(item)
    
    return web.json_response({"folders": sorted(folders)}, status=200)


@server.PromptServer.instance.routes.post("/prompt_viewer/generate_caption")
async def generate_caption(request):
    """
    API endpoint to generate a caption for an image using Moondream2
    """
    try:
        data = await request.json()
    except:
        return web.Response(text="Invalid JSON", status=400)
    
    folder = data.get("folder", "(root)")
    image_filename = data.get("image_filename", "")
    
    if not image_filename:
        return web.Response(text="No image filename provided", status=400)
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Build file path based on folder
    if folder == "(root)":
        image_path = os.path.join(prompts_dir, image_filename)
    else:
        image_path = os.path.join(prompts_dir, folder, image_filename)
    
    # Security check: ensure the file is within the prompts directory
    if not os.path.abspath(image_path).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    if not os.path.exists(image_path):
        return web.Response(text=f"Image not found: {image_filename}", status=404)
    
    # Check if it's an image file
    if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
        return web.Response(text="Not an image file", status=400)
    
    try:
        # Get node instance and generate caption
        node = get_gr_prompt_viewer_instance()
        if node is None:
            return web.Response(text="Failed to initialize caption generator", status=500)
        
        print(f"Generating caption for: {image_path}")
        caption = node.generate_caption_for_image(image_path)
        print(f"Caption generated: {caption[:100]}...")
        
        return web.json_response({"caption": caption}, status=200)
    except Exception as e:
        print(f"Error generating caption: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(text=f"Error generating caption: {str(e)}", status=500)

@server.PromptServer.instance.routes.post("/prompt_viewer/auto_save_caption")
async def auto_save_caption(request):
    """
    API endpoint to auto-save generated caption with same name as image file
    """
    try:
        data = await request.json()
    except:
        return web.Response(text="Invalid JSON", status=400)
    
    folder = data.get("folder", "(root)")
    image_filename = data.get("image_filename", "")
    caption = data.get("caption", "")
    
    if not image_filename:
        return web.Response(text="No image filename provided", status=400)
    
    if not caption or caption.strip() == "":
        return web.Response(text="No caption provided", status=400)
    
    # Get the custom node directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")
    
    # Determine target directory
    if folder == "(root)":
        target_dir = prompts_dir
    else:
        target_dir = os.path.join(prompts_dir, folder)
    
    # Security check
    if not os.path.abspath(target_dir).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid folder", status=403)
    
    # Create folder if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Create text filename from image filename
    # Remove image extension and add .txt
    base_name = os.path.splitext(image_filename)[0]
    txt_filename = f"{base_name}.txt"
    
    # Build full path for the text file
    txt_filepath = os.path.join(target_dir, txt_filename)
    
    # Security check again
    if not os.path.abspath(txt_filepath).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    try:
        # Save the caption to the text file
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        print(f"✓ Caption auto-saved: {txt_filename} in {folder}")
        
        return web.json_response({
            "message": "Caption auto-saved successfully",
            "filename": txt_filename,
            "folder": folder
        }, status=200)
    except Exception as e:
        print(f"✗ Error auto-saving caption: {e}")
        return web.Response(text=f"Error auto-saving caption: {str(e)}", status=500)



@server.PromptServer.instance.routes.get("/dynamic_lora/list_loras")
async def dynamic_lora_list(request):
    """
    Return JSON list of LoRA paths found under models/lora (recursive).
    Each path is relative to models/lora root, e.g. 'my_subdir/my_lora.safetensors'
    """
    # Find models/lora path (similar logic to the node)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_candidates = [
        os.path.abspath(os.path.join(current_dir, "..", "..")),
        os.path.abspath(os.path.join(current_dir, "..")),
        os.path.abspath(current_dir)
    ]
    models_lora_dir = None
    for cand in project_root_candidates:
        possible = os.path.join(cand, "models", "lora")
        if os.path.exists(possible):
            models_lora_dir = possible
            break

    # fallback: try local models/lora
    if models_lora_dir is None:
        possible = os.path.join(current_dir, "models", "lora")
        if os.path.exists(possible):
            models_lora_dir = possible

    if models_lora_dir is None:
        return web.json_response({"loras": []}, status=200)

    found = []
    for root, dirs, files in os.walk(models_lora_dir):
        for f in files:
            if f.lower().endswith((".safetensors", ".pt", ".bin")):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, models_lora_dir)
                found.append(rel.replace("\\", "/"))

    found.sort()
    return web.json_response({"loras": found}, status=200)


# Simple LoRA Stack API Routes
@server.PromptServer.instance.routes.get("/gr_lora_loader/loras")
async def get_gr_lora_loader_loras(request):
    """
    Returns a list of available LoRAs using ComfyUI's folder_paths
    """
    try:
        loras = folder_paths.get_filename_list("loras")
        return web.json_response(list(loras))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/gr_lora_loader/save_config")
async def save_lora_stack_config(request):
    """
    Save LoRA stack configuration to a JSON file
    """
    try:
        data = await request.json()
        node_id = data.get("node_id")
        config = data.get("config")
        
        # print(f"[GRLoraLoader] Save request received")
        # print(f"[GRLoraLoader] Node ID: {node_id}")
        # print(f"[GRLoraLoader] Config: {json.dumps(config, indent=2)}")
        
        if not node_id or not config:
            # print(f"[GRLoraLoader] ERROR: Missing node_id or config")
            return web.json_response({"error": "Missing node_id or config"}, status=400)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, "lora_configs")
        
        # print(f"[GRLoraLoader] Config directory: {config_dir}")
        
        if not os.path.exists(config_dir):
            # print(f"[GRLoraLoader] Creating config directory...")
            os.makedirs(config_dir)
        
        config_file = os.path.join(config_dir, f"GRLoraLoader_{node_id}.json")
        # print(f"[GRLoraLoader] Saving to file: {config_file}")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # print(f"[GRLoraLoader] ✓ Configuration saved successfully!")
        return web.json_response({"success": True, "message": "Configuration saved", "file": config_file})
    except Exception as e:
        # print(f"[GRLoraLoader] ✗ ERROR saving configuration: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/gr_lora_loader/load_config")
async def load_lora_stack_config(request):
    """
    Load LoRA stack configuration from a JSON file
    """
    try:
        node_id = request.query.get("node_id")
        
        # print(f"[GRLoraLoader] Load request received")
        # print(f"[GRLoraLoader] Node ID: {node_id}")
        
        if not node_id:
            # print(f"[GRLoraLoader] ERROR: Missing node_id")
            return web.json_response({"error": "Missing node_id"}, status=400)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "lora_configs", f"GRLoraLoader_{node_id}.json")
        
        # print(f"[GRLoraLoader] Looking for config file: {config_file}")
        
        if not os.path.exists(config_file):
            # print(f"[GRLoraLoader] Config file not found, returning empty config")
            return web.json_response({"config": None})
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # print(f"[GRLoraLoader] ✓ Configuration loaded successfully!")
        # print(f"[GRLoraLoader] Config: {json.dumps(config, indent=2)}")
        return web.json_response({"config": config})
    except Exception as e:
        # print(f"[GRLoraLoader] ✗ ERROR loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


print("✓ GRPromptViewer routes registered (including caption generation)")