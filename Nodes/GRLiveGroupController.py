import os
import json
import server
from aiohttp import web

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

SAVE_DIR  = os.path.dirname(__file__)
SAVE_FILE = os.path.join(SAVE_DIR, "gr_lgc_order.json")


# ── REST endpoints ────────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.get("/gr_lgc/load")
async def lgc_load(request):
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return web.json_response(data)


@server.PromptServer.instance.routes.post("/gr_lgc/save")
async def lgc_save(request):
    data = await request.json()
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return web.json_response({"ok": True})


# ── Node ──────────────────────────────────────────────────────────────────────

class GRLiveGroupController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def run(self, unique_id=None, extra_pnginfo=None):
        return {}


NODE_CLASS_MAPPINGS = {
    "GRLiveGroupController": GRLiveGroupController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLiveGroupController": "GR Live Group Controller",
}
