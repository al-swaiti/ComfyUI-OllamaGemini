import os
import json
import google.generativeai as genai
from PIL import Image
import requests
import torch
import codecs
from openai import OpenAI
import base64
import folder_paths
import anthropic
import io
import numpy as np
from .clipseg import CLIPSeg, CombineMasks
from .BRIA_RMBG import BRIA_RMBG_ModelLoader, BRIA_RMBG
from .svgnode import ConvertRasterToVector, SaveSVG
from .FLUXResolutions import FLUXResolutions
from .prompt_styler import *
from datetime import datetime

# ================== UNIVERSAL IMAGE UTILITIES ==================
def rgba_to_rgb(image):
    """Convert RGBA image to RGB with white background"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    return image

def tensor_to_pil_image(tensor):
    """Convert tensor to PIL Image with RGBA support"""
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    
    # Handle different channel counts
    if len(image_np.shape) == 2:  # Grayscale
        image_np = np.expand_dims(image_np, axis=-1)
    if image_np.shape[-1] == 1:   # Single channel
        image_np = np.repeat(image_np, 3, axis=-1)
        
    channels = image_np.shape[-1]
    mode = 'RGBA' if channels == 4 else 'RGB'
    
    image = Image.fromarray(image_np, mode=mode)
    return rgba_to_rgb(image)

def tensor_to_base64(tensor):
    """Convert tensor to base64 encoded PNG"""
    image = tensor_to_pil_image(tensor)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
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

# ================== API SERVICES ==================
import os
import json
from openai import OpenAI
import requests
import base64
import numpy as np
from PIL import Image
import io
import torch

class QwenAPI:
    def __init__(self):
        self.qwen_api_key = self.get_qwen_api_key()
        if not self.qwen_api_key:
            print("Error: Qwen API key is required")
        self.client = OpenAI(
            api_key=self.qwen_api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

    def get_qwen_api_key(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("QWEN_API_KEY", "")
        except Exception as e:
            print(f"Error loading Qwen API key: {str(e)}")
            return ""

    def tensor_to_base64(self, image_tensor):
        # Ensure the tensor is on CPU and convert to numpy
        if torch.is_tensor(image_tensor):
            if image_tensor.ndim == 4:
                image_tensor = image_tensor[0]
            image_tensor = (image_tensor * 255).clamp(0, 255)
            image_tensor = image_tensor.cpu().numpy().astype(np.uint8)
            if image_tensor.shape[0] == 3:  # If channels are first
                image_tensor = image_tensor.transpose(1, 2, 0)
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_tensor)
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "qwen_model": (
                    [
                        "qwen-max", "qwen-max-latest", "qwen-max-2025-01-25",
                        "qwen-plus", "qwen-plus-latest", "qwen-plus-2025-01-25",
                        "qwen-turbo", "qwen-turbo-latest", "qwen-turbo-2024-11-01",
                        "qwen-vl-max", "qwen-vl-plus",
                        "qwen2.5-vl-72b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct",
                        "qwen2.5-7b-instruct-1m", "qwen2.5-14b-instruct-1m", "qwen2.5-72b-instruct",
                        "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct",
                        "qwen2-72b-instruct", "qwen2-57b-a14b-instruct", "qwen2-7b-instruct",
                        "qwen1.5-110b-chat", "qwen1.5-7b-chat", "qwen1.5-72b-chat",
                        "qwen1.5-32b-chat", "qwen1.5-14b-chat",
                        "text-embedding-v3"
                    ],
                    {"default": "qwen-max"}
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Qwen"

    def generate_content(self, prompt, qwen_model, max_tokens, temperature, top_p, image=None):
        if not self.qwen_api_key:
            return ("Qwen API key missing",)

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            if image is not None:
                image_b64 = self.tensor_to_base64(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=qwen_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Extract the response text from the completion
            response_text = completion.choices[0].message.content
            return (response_text,)

        except Exception as e:
            return (f"API Error: {str(e)}",)
class OpenAIAPI:
    def __init__(self):
        self.openai_api_key = self.get_openai_api_key()
        self.nvidia_api_key = self.get_nvidia_api_key()
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        if self.nvidia_api_key:
            self.nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.nvidia_api_key
            )
    
    def get_openai_api_key(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:  
                config = json.load(f)
            return config["OPENAI_API_KEY"]
        except:
            print("Error: OpenAI API key is required")
            return ""

    def get_nvidia_api_key(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:  
                config = json.load(f)
            return config.get("NVIDIA_API_KEY")
        except:
            print("Error: NVIDIA API key is required")
            return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "model": ([
                    # GPT-4 Family
                    "gpt-4o-mini",
                    "gpt-4o-mini-2024-07-18",
                    # GPT-3.5 Family
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-instruct",
                    "gpt-3.5-turbo-instruct-0914",
                    # O1 Family
                    "o1-preview",
                    "o1-preview-2024-09-12",
                    "o1-mini",
                    "o1-mini-2024-09-12",
                    # DeepSeek Models
                    "deepseek-ai/deepseek-r1",
                    # Legacy Models
                    "babbage-002",
                    "davinci-002"
                ],),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "stream": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/OpenAI"

    def generate_content(self, prompt, model, max_tokens, temperature, top_p, stream, image=None):
        messages = [{"role": "user", "content": prompt}]
        
        if image is not None:
            image_b64 = tensor_to_base64(image)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]

        try:
            client = self.nvidia_client if model.startswith("deepseek") else self.openai_client
            if not client:
                raise ValueError("API client not initialized")

            generation_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

            if stream:
                response = client.chat.completions.create(**generation_params, stream=True)
                textoutput = "".join([chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content])
            else:
                response = client.chat.completions.create(**generation_params)
                textoutput = response.choices[0].message.content

        except Exception as e:
            textoutput = f"API Error: {str(e)}"

        return (textoutput,)

class ClaudeAPI:
    def __init__(self):
        self.claude_api_key = self.get_claude_api_key()
        if self.claude_api_key:
            self.client = anthropic.Client(api_key=self.claude_api_key)
    
    def get_claude_api_key(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:  
                config = json.load(f)
            return config["CLAUDE_API_KEY"]
        except:
            print("Error: Claude API key is required")
            return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Claude"

    def generate_content(self, prompt, model, max_tokens, image=None):
        if not self.claude_api_key:
            return ("Claude API key missing",)

        messages = [{"role": "user", "content": prompt}]
        
        try:
            if image is not None:
                image_b64 = tensor_to_base64(image)
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
                ]

            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            return (response.content[0].text,)
        except Exception as e:
            return (f"API Error: {str(e)}",)

class GeminiAPI:
    def __init__(self):
        self.gemini_api_key = get_gemini_api_key()
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "gemini_model": ([
                    # Gemini 2.0 Models
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite-preview-02-05",
                    "gemini-2.0-pro-exp-02-05",
                    "gemini-2.0-flash-thinking-exp-01-21",
                    "gemini-2.0-flash-exp",
                    # Gemini 1.5 Models
                    "gemini-1.5-pro",
                    "gemini-1.5-flash-8b",
                    "gemini-1.5-pro-experimental",
                    "learnlm-1.5-pro-experimental",
                    # Gemma Models
                    "gemma-2-2b-it",
                    "gemma-2-9b-it",
                    "gemma-2-27b-it"
                ],),
                "stream": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),  
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Gemini"

    def generate_content(self, prompt, gemini_model, stream, image=None):
        if not self.gemini_api_key:
            return ("Gemini API key missing",)

        try:
            model = genai.GenerativeModel(gemini_model)
            content = [prompt]

            if image is not None:
                pil_image = tensor_to_pil_image(image)
                content.append(pil_image)

            if stream:
                response = model.generate_content(content, stream=True)
                textoutput = "\n".join([chunk.text for chunk in response])
            else:
                response = model.generate_content(content)
                textoutput = response.text

            return (textoutput,)
        except Exception as e:
            return (f"API Error: {str(e)}",)

class OllamaAPI:
    def __init__(self):
        self.ollama_url = get_ollama_url()

    @classmethod
    def get_ollama_models(cls):
        ollama_url = get_ollama_url()
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return ["llama2"]
        except Exception as e:
            print(f"Error fetching Ollama models: {str(e)}")
            return ["llama2"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "ollama_model": (cls.get_ollama_models(),),
                "keep_alive": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),                
            },
            "optional": {
                "image": ("IMAGE",),  
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Ollama"

    def generate_content(self, prompt, ollama_model, keep_alive, image=None):
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": f"{keep_alive}m"
        }

        try:
            if image is not None:
                pil_image = tensor_to_pil_image(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                payload["images"] = [base64.b64encode(buffered.getvalue()).decode()]

            response = requests.post(url, json=payload)
            response.raise_for_status()
            return (response.json().get('response', ''),)
        except Exception as e:
            return (f"API Error: {str(e)}",)

# ================== SUPPORTING NODES ==================
class TextSplitByDelimiter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,"dynamicPrompts": False}),
                "delimiter":("STRING", {"multiline": False,"default":",","dynamicPrompts": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "skip_every": ("INT", {"default": 0, "min": 0, "max": 10}),
                "max_count": ("INT", {"default": 10, "min": 1, "max": 1000}),
            }
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "AI API"

    def run(self, text, delimiter, start_index, skip_every, max_count):
        delimiter = codecs.decode(delimiter, 'unicode_escape')
        arr = [item.strip() for item in text.split(delimiter) if item.strip()]
        arr = arr[start_index:start_index + max_count * (skip_every + 1):(skip_every + 1)]
        return (arr,)

class Save_text_File:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": 'info', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "text": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"
    CATEGORY = "AI API"

    def save_text_file(self, text="", path="", filename=""):
        output_path = os.path.join(self.output_dir, path)
        os.makedirs(output_path, exist_ok=True)
        
        if not filename:
            filename = datetime.now().strftime('%Y%m%d%H%M%S')
            
        file_path = os.path.join(output_path, f"{filename}.txt")
        try:
            with open(file_path, 'w') as f:
                f.write(text)
        except OSError:
            print(f'Error saving file: {file_path}')
            
        return (text,)

# ================== NODE REGISTRATION ==================
NODE_CLASS_MAPPINGS = {
    "CLIPSeg": CLIPSeg,
    "QwenAPI": QwenAPI,
    "CombineSegMasks": CombineMasks,
    "OpenAIAPI": OpenAIAPI,
    "ClaudeAPI": ClaudeAPI,
    "GeminiAPI": GeminiAPI,
    "OllamaAPI": OllamaAPI,
    "TextSplitByDelimiter": TextSplitByDelimiter,
    "Save text": Save_text_File,
    "BRIA_RMBG_ModelLoader": BRIA_RMBG_ModelLoader,
    "BRIA_RMBG": BRIA_RMBG,
    "ConvertRasterToVector": ConvertRasterToVector,
    "SaveSVG": SaveSVG,
    "FLUXResolutions": FLUXResolutions,
    'ComfyUIStyler': type('ComfyUIStyler', (PromptStyler,), {'menus': NODES['ComfyUI Styler']})
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSeg": "CLIPSeg",
    "QwenAPI": "Qwen API",
    "CombineSegMasks": "CombineMasks",
    "OpenAIAPI": "OpenAI API",
    "ClaudeAPI": "Claude API",
    "GeminiAPI": "Gemini API",
    "OllamaAPI": "Ollama API",
    "TextSplitByDelimiter": "TextSplitByDelimiter",
    "Save text": "Save_text_File",
    "BRIA_RMBG_ModelLoader": "BRIA_RMBG Model Loader",
    "BRIA_RMBG": "BRIA RMBG",
    "ConvertRasterToVector": "Raster to Vector (SVG)",
    "SaveSVG": "Save SVG",
    "FLUXResolutions": "FLUX Resolutions",
    'ComfyUIStyler': 'ComfyUI Styler'
}