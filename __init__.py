import os
import json
from pathlib import Path

# Define paths
current_dir = Path(__file__).parent
config_file = current_dir / "config.json"

# Create or load config
if not config_file.exists():
    config = {
        "GEMINI_API_KEY": "your_gemini_api_key_here",
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "OLLAMA_URL": "http://localhost:11434"
    }
    config_file.write_text(json.dumps(config, indent=4))
    print("Created config.json with default values. Please update with your API keys.")
else:
    config = json.loads(config_file.read_text())

# Import node mappings from renamed node classes
from .GeminiOllamaNode import NODE_CLASS_MAPPINGS as GEMINI_OLLAMA_MAPPINGS
from .GeminiOllamaNode import NODE_DISPLAY_NAME_MAPPINGS as GEMINI_OLLAMA_DISPLAY_MAPPINGS

# Import node mappings from image generator nodes
from .GeminiImageGenerationNode import NODE_CLASS_MAPPINGS as GEMINI_IMAGE_MAPPINGS
from .GeminiImageGenerationNode import NODE_DISPLAY_NAME_MAPPINGS as GEMINI_IMAGE_DISPLAY_MAPPINGS

# Import Veo Video Generation nodes
from .VeoVideoGenerationNode import NODE_CLASS_MAPPINGS as VEO_VIDEO_MAPPINGS
from .VeoVideoGenerationNode import NODE_DISPLAY_NAME_MAPPINGS as VEO_VIDEO_DISPLAY_MAPPINGS

# Import Smart Prompt Generator node
from .GeminiSmartPromptGenerator import NODE_CLASS_MAPPINGS as SMART_PROMPT_MAPPINGS
from .GeminiSmartPromptGenerator import NODE_DISPLAY_NAME_MAPPINGS as SMART_PROMPT_DISPLAY_MAPPINGS

# Import model listing functionality
from .list_models import get_gemini_models, get_openai_models, get_gemini_image_models

# Import from BRIA_RMBG
from .BRIA_RMBGx import GeminiBRIA_RMBG

# Import from clipseg
from .clipsegx import GeminiCLIPSeg, GeminiCombineSegMasks

# Import from svgnode
from .svgnodex import GeminiConvertRasterToVector, GeminiSaveSVG, GeminiSVGPreview

# Import from FLUXResolutions
from .FLUXResolutions import GeminiFLUXResolutions

# GeminiTextToPrompt module is missing, creating empty mappings
GEMINI_PROMPT_MAPPINGS = {}
GEMINI_PROMPT_DISPLAY_MAPPINGS = {}

# SimpleGeminiGenerator module is missing, creating empty mappings
SIMPLE_GEMINI_MAPPINGS = {}
SIMPLE_GEMINI_DISPLAY_MAPPINGS = {}

# Structured template nodes removed
STRUCTURED_TEMPLATE_MAPPINGS = {}
STRUCTURED_TEMPLATE_DISPLAY_MAPPINGS = {}

# Import prompt styler nodes
from .prompt_stylerx import NODES, PromptStyler

# Combine node mappings
NODE_CLASS_MAPPINGS = {
    **GEMINI_OLLAMA_MAPPINGS,
    **GEMINI_IMAGE_MAPPINGS,
    **VEO_VIDEO_MAPPINGS,
    **GEMINI_PROMPT_MAPPINGS,
    **SIMPLE_GEMINI_MAPPINGS,
    **STRUCTURED_TEMPLATE_MAPPINGS,
    **SMART_PROMPT_MAPPINGS,

    # Additional nodes - renamed to avoid conflicts
    "GeminiBRIA_RMBG": GeminiBRIA_RMBG,
    "GeminiCLIPSeg": GeminiCLIPSeg,
    "GeminiCombineSegMasks": GeminiCombineSegMasks, # Renamed from GeminiCombineMasks
    "GeminiConvertRasterToVector": GeminiConvertRasterToVector, # Renamed
    "GeminiSaveSVG": GeminiSaveSVG,
    "GeminiSVGPreview": GeminiSVGPreview,
    "GeminiFLUXResolutions": GeminiFLUXResolutions, # Renamed
    
    # Prompt styler nodes
    'GeminiComfyUIStyler': type('GeminiComfyUIStyler', (PromptStyler,), {'style_menus': NODES['Gemini ComfyUI Styler']})
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **GEMINI_OLLAMA_DISPLAY_MAPPINGS,
    **GEMINI_IMAGE_DISPLAY_MAPPINGS,
    **VEO_VIDEO_DISPLAY_MAPPINGS,
    **GEMINI_PROMPT_DISPLAY_MAPPINGS,
    **SIMPLE_GEMINI_DISPLAY_MAPPINGS,
    **STRUCTURED_TEMPLATE_DISPLAY_MAPPINGS,
    **SMART_PROMPT_DISPLAY_MAPPINGS,

    # Additional nodes - renamed to avoid conflicts
    "GeminiBRIA_RMBG": "Gemini BRIA RMBG",
    "GeminiCLIPSeg": "Gemini CLIPSeg",
    "GeminiCombineSegMasks": "Gemini Combine Seg Masks", # Renamed from GeminiCombineMasks
    "GeminiConvertRasterToVector": "Gemini Convert Raster to Vector", # Renamed
    "GeminiSaveSVG": "Gemini Save SVG",
    "GeminiSVGPreview": "Gemini SVG Preview",
    "GeminiFLUXResolutions": "Gemini FLUX Resolutions", # Renamed
    
    # Prompt styler nodes
    'GeminiComfyUIStyler': 'Gemini ComfyUI Styler'
}

# Web directory for custom JavaScript widgets (video preview, upload, etc.)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']