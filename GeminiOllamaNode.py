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
from .clipseg import CLIPSeg, CombineMasks
from .BRIA_RMBG import BRIA_RMBG_ModelLoader, BRIA_RMBG
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
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-1106",
                    "o1-preview",
                    "o1-mini",  # Latest GPT-3.5 Turbo
                    "deepseek-ai/deepseek-r1"  # NVIDIA model
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

    def tensor_to_base64(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate_content(self, prompt, model, max_tokens, temperature, top_p, stream, image=None):
        messages = [{"role": "user", "content": prompt}]
        
        if image is not None:
            image_b64 = self.tensor_to_base64(image)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]

        # Determine which client to use based on the model
        if model.startswith("deepseek"):
            if not hasattr(self, 'nvidia_client') or not self.nvidia_api_key:
                raise ValueError("NVIDIA API key is required for NVIDIA models")
            client = self.nvidia_client
        else:
            if not hasattr(self, 'openai_client') or not self.openai_api_key:
                raise ValueError("OpenAI API key is required")
            client = self.openai_client

        # Prepare generation parameters
        generation_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        # Handle streaming and non-streaming responses
        if stream:
            response = client.chat.completions.create(**generation_params, stream=True)
            textoutput = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    textoutput += chunk.choices[0].delta.content
        else:
            response = client.chat.completions.create(**generation_params)
            textoutput = response.choices[0].message.content
        
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
            api_key = config["CLAUDE_API_KEY"]
        except:
            print("Error: Claude API key is required")
            return ""
        return api_key

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

    def tensor_to_base64(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate_content(self, prompt, model, max_tokens, image=None):
        if not self.claude_api_key:
            raise ValueError("Claude API key is required")

        messages = [{"role": "user", "content": prompt}]
        
        if image is not None:
            image_b64 = self.tensor_to_base64(image)
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
                "gemini_model": (["gemini-1.5-pro-002", "gemini-1.5-flash", "gemini-1.5-flash-8b","learnlm-1.5-pro-experimental","gemini-exp-1114","gemini-exp-1121"],),
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

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def generate_content(self, prompt, gemini_model, stream, image=None):
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        
        model = genai.GenerativeModel(gemini_model)

        if gemini_model in ["gemini-1.5-pro-002", "gemini-1.5-flash", "gemini-1.5-flash-8b","learnlm-1.5-pro-experimental","gemini-exp-1114","gemini-exp-1121"]:
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

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def generate_content(self, prompt, ollama_model, keep_alive, image=None):
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": f"{keep_alive}m"
        }

        if image is not None and isinstance(image, torch.Tensor) and image.numel() > 0:
            pil_image = self.tensor_to_image(image)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            payload["images"] = [img_str]

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


import os
from datetime import datetime

class Save_text_File:
    """
    This class is responsible for saving text content to a file.

    It provides a standardized way to save text data, ensuring that the necessary directories are created if they don't exist, and handling empty or missing input data gracefully.
    """

    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types and default values for the class.

        Returns:
            dict: A dictionary with the required input parameters and their types/defaults.
        """
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
        """
        Saves the provided text content to a file.

        If the specified output path doesn't exist, it will create the necessary directories.
        If the filename is empty, it will use a timestamp-based filename.
        If the positive text is empty, it will save a default message.

        Args:
            positive (str): The text content to be saved.
            path (str): The relative path where the file should be saved.
            filename (str): The name of the file to be saved (without the .txt extension).

        Returns:
            tuple: A tuple containing the saved text content.
        """
        output_path = os.path.join(self.output_dir, path)

        # Check if the output path exists, and create it if it doesn't
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        # If the filename is empty, use a timestamp-based filename
        if filename.strip() == '':
            print(f'Warning: There is no filename specified! Saving file with timestamp.')
            filename = get_timestamp('%Y%m%d%H%M%S')

        # If the positive text is empty, use a default message
        if text == "":
            text = "No prompt data"

        # Save the text content to the file
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), text)
        return (text,)

    def writeTextFile(self, file, content):
        """
        Writes the provided content to the specified file.

        Args:
            file (str): The full path and filename of the file to be written.
            content (str): The text content to be written to the file.
        """
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'Error: Unable to save file `{file}`')

def get_timestamp(fmt):
    """
    Generates a timestamp string based on the provided format.

    Args:
        fmt (str): The format string for the timestamp.

    Returns:
        str: The timestamp string.
    """
    return datetime.now().strftime(fmt)


NODE_CLASS_MAPPINGS = {
    "CLIPSeg": CLIPSeg,
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