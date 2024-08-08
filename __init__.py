import os
import json
from pathlib import Path

# Define paths
current_dir = Path(__file__).parent
config_file = current_dir / "config.json"

# Create or load config
if not config_file.exists():
    config = {
        "GEMINI_API_KEY": "your key",
        "OLLAMA_URL": "http://localhost:11434"
    }
    config_file.write_text(json.dumps(config, indent=4))
    print("Created config.json with default values")
else:
    config = json.loads(config_file.read_text())

# Import node mappings
from .GeminiOllamaNode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']