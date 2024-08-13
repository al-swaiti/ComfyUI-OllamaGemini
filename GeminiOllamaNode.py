import os
import json
import google.generativeai as genai
from PIL import Image
import requests
import torch
import codecs
from  .BRIA_RMBG import BRIA_RMBG_ModelLoader, BRIA_RMBG
from .svgnode import ConvertRasterToVector, SaveSVG
from .FLUXResolutions import FLUXResolutions
from .prompt_styler import *

def get_gemini_api_key():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        api_key = config["GEMINI_API_KEY"]
    except:
        print("Error: Gemini API key is required")
        return ""
    return api_key

def get_ollama_url():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        ollama_url = config.get("OLLAMA_URL", "http://localhost:11434")
    except:
        print("Error: Ollama URL not found, using default")
        ollama_url = "http://localhost:11434"
    return ollama_url

class GeminiOllamaAPI:

    def __init__(self):
        self.gemini_api_key = get_gemini_api_key()
        self.ollama_url = get_ollama_url()
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key, transport='rest')

    @classmethod
    def get_ollama_models(cls):
        ollama_url = get_ollama_url()
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                print(f"Failed to fetch Ollama models. Status code: {response.status_code}")
                return ["llama2"]  # Fallback to a default model
        except Exception as e:
            print(f"Error fetching Ollama models: {str(e)}")
            return ["llama2"]  # Fallback to a default model

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_choice": (["Gemini", "Ollama"],),
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "gemini_model": (["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro-latest", "gemini-1.5-pro-exp-0801", "gemini-1.5-flash"],),
                "ollama_model": (cls.get_ollama_models(),),
                "stream": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),  
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"

    CATEGORY = "AI API"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def generate_content(self, api_choice, prompt, gemini_model, ollama_model, stream, image=None):
        if api_choice == "Gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API key is required")
            return self.generate_gemini_content(prompt, gemini_model, stream, image)
        elif api_choice == "Ollama":
            return self.generate_ollama_content(prompt, ollama_model, stream, image)

    def generate_gemini_content(self, prompt, model_name, stream, image=None):
        model = genai.GenerativeModel(model_name)

        if model_name in ['gemini-1.5-pro-latest', 'gemini-1.5-pro-exp-0801', 'gemini-1.5-flash']:
            if image is None:
                if stream:
                    response = model.generate_content(prompt, stream=True)
                    textoutput = "\n".join([chunk.text for chunk in response])
                else:
                    response = model.generate_content(prompt)
                    textoutput = response.text
            else:
                pil_image = self.tensor_to_image(image)
                if stream:
                    response = model.generate_content([prompt, pil_image], stream=True)
                    textoutput = "\n".join([chunk.text for chunk in response])
                else:
                    response = model.generate_content([prompt, pil_image])
                    textoutput = response.text
        

        return (textoutput,)

    def generate_ollama_content(self, prompt, model_name, stream, image=None):
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }

        if image is not None and isinstance(image, torch.Tensor) and image.numel() > 0:
            pil_image = self.tensor_to_image(image)
            # Convert PIL image to base64
            import base64
            import io
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            payload["images"] = [img_str]

        if stream:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            textoutput = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data = json.loads(decoded_line[6:])
                        textoutput += data.get('response', '')
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            textoutput = response.json().get('response', '')

        return (textoutput,)


class TextSplitByDelimiter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,"dynamicPrompts": False}),
                "delimiter":("STRING", {"multiline": False,"default":",","dynamicPrompts": False}),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 1000, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                 "skip_every": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 10, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "max_count": ("INT", {
                    "default": 10,
                    "min": 1, #Minimum value
                    "max": 1000, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
            }
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "AI API"

    def run(self, text, delimiter, start_index, skip_every, max_count):
        if delimiter == "":
            arr = [text.strip()]
        else:
            delimiter = codecs.decode(delimiter, 'unicode_escape')
            arr = [item.strip() for item in text.split(delimiter) if item.strip()]
        
        arr = arr[start_index:start_index + max_count * (skip_every + 1):(skip_every + 1)]
        
        return (arr,)


NODE_CLASS_MAPPINGS = {
    "GeminiOllamaAPI": GeminiOllamaAPI,
    "TextSplitByDelimiter": TextSplitByDelimiter,
    "BRIA_RMBG_ModelLoader": BRIA_RMBG_ModelLoader,
    "BRIA_RMBG": BRIA_RMBG,
    "ConvertRasterToVector": ConvertRasterToVector,
    "SaveSVG": SaveSVG,
    "FLUXResolutions": FLUXResolutions,
    'ComfyUIStyler': type('ComfyUIStyler', (PromptStyler,), {'menus': NODES['ComfyUI Styler']})
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiOllamaAPI": "Gemini Ollama API",
    "TextSplitByDelimiter": "TextSplitByDelimiter",
    "BRIA_RMBG_ModelLoader": "BRIA_RMBG Model Loader",
    "BRIA_RMBG": "BRIA RMBG",
    "ConvertRasterToVector": "Raster to Vector (SVG)",
    "SaveSVG": "Save SVG",
    "FLUXResolutions": "FLUX Resolutions",
    'ComfyUIStyler': 'ComfyUI Styler'
}