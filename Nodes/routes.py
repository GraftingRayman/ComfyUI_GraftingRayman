import os
import json
import server
from aiohttp import web

@server.PromptServer.instance.routes.get("/prompt_viewer/list_files")
async def list_files(request):
    """
    API endpoint to list files in a specific folder
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
        # Get list of text files in the folder
        files = [f for f in os.listdir(target_dir) 
                if os.path.isfile(os.path.join(target_dir, f)) 
                and f.endswith(('.txt', '.log', '.json', '.csv', '.md'))]
        
        if not files:
            files = ["No files found"]
        else:
            files = sorted(files)
        
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
    
    # Build file path based on folder
    if folder == "(root)":
        file_path = os.path.join(prompts_dir, filename)
    else:
        folder_path = os.path.join(prompts_dir, folder)
        file_path = os.path.join(folder_path, filename)
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # Security check: ensure the file is within the prompts directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(prompts_dir)):
        return web.Response(text="Invalid file path", status=403)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return web.json_response({"message": "File saved successfully", "filename": filename}, status=200)
    except Exception as e:
        return web.Response(text=f"Error saving file: {str(e)}", status=500)


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