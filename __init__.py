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

# Import node mappings from original node
from .GeminiOllamaNode import NODE_CLASS_MAPPINGS as GEMINI_OLLAMA_MAPPINGS
from .GeminiOllamaNode import NODE_DISPLAY_NAME_MAPPINGS as GEMINI_OLLAMA_DISPLAY_MAPPINGS

# Import node mappings from image generator nodes
from .GeminiImageGenerationNode import NODE_CLASS_MAPPINGS as GEMINI_IMAGE_MAPPINGS
from .GeminiImageGenerationNode import NODE_DISPLAY_NAME_MAPPINGS as GEMINI_IMAGE_DISPLAY_MAPPINGS

# Import Smart Prompt Generator node
from .SmartPromptGenerator import NODE_CLASS_MAPPINGS as SMART_PROMPT_MAPPINGS
from .SmartPromptGenerator import NODE_DISPLAY_NAME_MAPPINGS as SMART_PROMPT_DISPLAY_MAPPINGS

# Import model listing functionality
from .list_models import get_gemini_models, get_openai_models, get_gemini_image_models

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
from .prompt_styler import NODES, PromptStyler

# Combine node mappings
NODE_CLASS_MAPPINGS = {
    **GEMINI_OLLAMA_MAPPINGS,
    **GEMINI_IMAGE_MAPPINGS,
    **GEMINI_PROMPT_MAPPINGS,
    **SIMPLE_GEMINI_MAPPINGS,
    **STRUCTURED_TEMPLATE_MAPPINGS,
    **SMART_PROMPT_MAPPINGS,

    # Prompt styler nodes
    'ComfyUIStyler': type('ComfyUIStyler', (PromptStyler,), {'menus': NODES['ComfyUI Styler']})
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **GEMINI_OLLAMA_DISPLAY_MAPPINGS,
    **GEMINI_IMAGE_DISPLAY_MAPPINGS,
    **GEMINI_PROMPT_DISPLAY_MAPPINGS,
    **SIMPLE_GEMINI_DISPLAY_MAPPINGS,
    **STRUCTURED_TEMPLATE_DISPLAY_MAPPINGS,
    **SMART_PROMPT_DISPLAY_MAPPINGS,

    # Prompt styler nodes
    'ComfyUIStyler': 'ComfyUI Styler'
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']