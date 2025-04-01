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
from .BRIA_RMBG import  BRIA_RMBG
from .svgnode import ConvertRasterToVector, SaveSVG
from .FLUXResolutions import FLUXResolutions
from .prompt_styler import *

# Common function to apply prompt structure templates
def apply_prompt_template(prompt, prompt_structure="Custom"):
    # Define prompt structure templates
    prompt_templates = {
        "HunyuanVideo": "Create a single cohesive paragraph prompt for HunyuanVideo based on my description. Include the subject, setting, action, camera movements, and style. Return ONLY the prompt text itself.",

        "Wan2.1": "Create a single concise paragraph prompt for Wan2.1 based on my description. Include the subject, setting, action, camera movements, and style. Return ONLY the prompt text itself.",

        "FLUX.1-dev": "Create a detailed paragraph prompt for FLUX.1-dev based on my description. Include subject, artistic style, depth effects, camera details, and lighting. Return ONLY the prompt text itself.",

        "SDXL": "Create a comma-separated tag prompt for SDXL based on my description. Include subject, medium, art style, lighting, environment, camera settings, and artist influences. Return ONLY the prompt text itself."
    }

    # Apply template based on prompt_structure parameter
    modified_prompt = prompt
    if prompt_structure != "Custom" and prompt_structure in prompt_templates:
        template = prompt_templates[prompt_structure]
        print(f"Applying {prompt_structure} template")
        modified_prompt = f"{prompt}\n\n{template}"
    else:
        # Fallback to checking if prompt contains a template request
        for template_name, template in prompt_templates.items():
            if template_name.lower() in prompt.lower():
                print(f"Detected {template_name} template request in prompt")
                modified_prompt = f"{prompt}\n\n{template}"
                break

    return modified_prompt
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
                        # Qwen Max/Plus/Turbo Models
                        "qwen-max",
                        "qwen-plus",
                        "qwen-turbo",
                        # Qwen Vision Models
                        "qwen-vl-max",
                        "qwen-vl-plus",
                        # Qwen 1.5 Models
                        "qwen1.5-32b-chat"
                    ],
                    {"default": "qwen-max"}
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "SDXL"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Qwen"

    def generate_content(self, prompt, qwen_model, max_tokens, temperature, top_p, structure_output, prompt_structure, structure_format, output_format, image=None):
        if not self.qwen_api_key:
            return ("Qwen API key missing",)

        try:
            # Apply prompt template
            modified_prompt = apply_prompt_template(prompt, prompt_structure)

            # Add structure format if requested
            if structure_output:
                print(f"Requesting structured output from {qwen_model}")
                # Add the structure format to the prompt
                modified_prompt = f"{modified_prompt}\n\n{structure_format}"
                print(f"Modified prompt with structure format")

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            if image is not None:
                image_b64 = self.tensor_to_base64(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": modified_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": modified_prompt})

            # Configure the request parameters
            request_params = {
                "model": qwen_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }



            print(f"Sending request to Qwen API with model: {qwen_model}")
            completion = self.client.chat.completions.create(**request_params)

            # Get the response text
            textoutput = completion.choices[0].message.content

            # Process the output based on the selected format
            if textoutput.strip():
                # Clean up the text output
                clean_text = textoutput.strip()

                # Remove any markdown code blocks if present
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        # Extract content between the first set of backticks
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            # Skip the language identifier line
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()

                # Remove any quotes around the text
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()

                # Remove any "Prompt:" or similar prefixes
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break

                # Format as JSON if requested
                if output_format == "json":
                    try:
                        # Create a JSON object with the appropriate key based on the prompt structure
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"

                        json_output = json.dumps({
                            key_name: clean_text
                        }, indent=2)

                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
                    # Just return the clean text
                    textoutput = clean_text
                    print("Returning raw text output")

            return (textoutput,)

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
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "SDXL"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/OpenAI"

    def generate_content(self, prompt, model, max_tokens, temperature, top_p, stream, structure_output, prompt_structure, structure_format, output_format, image=None):
        # Apply prompt template
        modified_prompt = apply_prompt_template(prompt, prompt_structure)

        # Add structure format if requested
        if structure_output:
            print(f"Requesting structured output from {model}")
            # Add the structure format to the prompt
            modified_prompt = f"{modified_prompt}\n\n{structure_format}"
            print(f"Modified prompt with structure format")

        messages = [{"role": "user", "content": modified_prompt}]

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

                # Process the output based on the selected format
                if textoutput.strip():
                    # Clean up the text output
                    clean_text = textoutput.strip()

                    # Remove any markdown code blocks if present
                    if clean_text.startswith("```") and "```" in clean_text[3:]:
                        first_block_end = clean_text.find("```", 3)
                        if first_block_end > 3:
                            # Extract content between the first set of backticks
                            language_line_end = clean_text.find("\n", 3)
                            if language_line_end > 3 and language_line_end < first_block_end:
                                # Skip the language identifier line
                                clean_text = clean_text[language_line_end+1:first_block_end].strip()
                            else:
                                clean_text = clean_text[3:first_block_end].strip()

                    # Remove any quotes around the text
                    if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                       (clean_text.startswith("'") and clean_text.endswith("'")):
                        clean_text = clean_text[1:-1].strip()

                    # Remove any "Prompt:" or similar prefixes
                    prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                    for prefix in prefixes_to_remove:
                        if clean_text.startswith(prefix):
                            clean_text = clean_text[len(prefix):].strip()
                            break

                    # Format as JSON if requested
                    if output_format == "json":
                        try:
                            # Create a JSON object with the appropriate key based on the prompt structure
                            key_name = "prompt"
                            if prompt_structure != "Custom":
                                key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"

                            json_output = json.dumps({
                                key_name: clean_text
                            }, indent=2)

                            print(f"Formatted output as JSON with key: {key_name}")
                            textoutput = json_output
                        except Exception as e:
                            print(f"Error formatting output as JSON: {str(e)}")
                    else:
                        # Just return the clean text
                        textoutput = clean_text
                        print("Returning raw text output")

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
                "model": ([
                    # Most intelligent model
                    "claude-3-7-sonnet-20250219",
                    # Fastest model for daily tasks
                    "claude-3-5-haiku-20241022",
                    # Excels at writing and complex tasks
                    "claude-3-opus-20240229",
                    # Additional models
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-haiku-20240307"
                ],),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "SDXL"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Claude"

    def generate_content(self, prompt, model, max_tokens, structure_output, prompt_structure, structure_format, output_format, image=None):
        if not self.claude_api_key:
            return ("Claude API key missing",)

        # Apply prompt template
        modified_prompt = apply_prompt_template(prompt, prompt_structure)

        # Add structure format if requested
        if structure_output:
            print(f"Requesting structured output from {model}")
            # Add the structure format to the prompt
            modified_prompt = f"{modified_prompt}\n\n{structure_format}"
            print(f"Modified prompt with structure format")

        messages = [{"role": "user", "content": modified_prompt}]

        try:
            if image is not None:
                image_b64 = tensor_to_base64(image)
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
                ]

            # Configure the request parameters
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages
            }



            print(f"Sending request to Claude API with {len(messages)} messages")
            response = self.client.messages.create(**request_params)

            # Get the response text
            textoutput = response.content[0].text

            # Process the output based on the selected format
            if textoutput.strip():
                # Clean up the text output
                clean_text = textoutput.strip()

                # Remove any markdown code blocks if present
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        # Extract content between the first set of backticks
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            # Skip the language identifier line
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()

                # Remove any quotes around the text
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()

                # Remove any "Prompt:" or similar prefixes
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break

                # Format as JSON if requested
                if output_format == "json":
                    try:
                        # Create a JSON object with the appropriate key based on the prompt structure
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"

                        json_output = json.dumps({
                            key_name: clean_text
                        }, indent=2)

                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
                    # Just return the clean text
                    textoutput = clean_text
                    print("Returning raw text output")

            return (textoutput,)
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
                    # Gemini 2.5 Models
                    "gemini-2.5-pro-exp-03-25",
                    # Gemini 2.0 Models
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    # Gemini 1.5 Models
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-8b"
                ],),
                "stream": ("BOOLEAN", {"default": False}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "SDXL"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),

            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Gemini"

    def generate_content(self, prompt, gemini_model, stream, structure_output, prompt_structure, structure_format, output_format, image=None):
        if not self.gemini_api_key:
            return ("Gemini API key missing",)

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }

            # We'll use the common apply_prompt_template function instead of defining templates here

            # Apply prompt template
            modified_prompt = apply_prompt_template(prompt, prompt_structure)

            # Add JSON structure format if requested
            if structure_output:
                print(f"Requesting structured output from {gemini_model}")
                # Add the structure format to the prompt
                modified_prompt = f"{modified_prompt}\n\n{structure_format}"
                print(f"Modified prompt with structure format")

            # Create the model and content
            model = genai.GenerativeModel(gemini_model)
            content = [modified_prompt]

            if image is not None:
                print("Processing image for Gemini API")
                pil_image = tensor_to_pil_image(image)
                content.append(pil_image)

            print(f"Sending request to Gemini API with model: {gemini_model}")
            try:
                if stream:
                    response = model.generate_content(content, generation_config=generation_config, stream=True)
                    textoutput = "\n".join([chunk.text for chunk in response])
                else:
                    # Set safety settings to be more permissive
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]

                    response = model.generate_content(
                        content,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )

                    if not hasattr(response, 'text'):
                        # Handle empty response
                        if hasattr(response, 'prompt_feedback'):
                            return (f"API Error: Content blocked - {response.prompt_feedback}",)
                        else:
                            return (f"API Error: Empty response from Gemini API",)

                    textoutput = response.text
            except Exception as e:
                print(f"Error generating content: {str(e)}")
                return (f"API Error: {str(e)}",)

            print("Gemini API response received successfully")

            # If structured output was requested, verify it's valid JSON
            if structure_output and textoutput.strip():
                try:
                    # Try to find JSON in the response
                    json_start = textoutput.find('{')
                    json_end = textoutput.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        json_text = textoutput[json_start:json_end+1]
                        # Try to parse the JSON to verify it's valid
                        json.loads(json_text)
                        print("Received valid JSON response from Gemini")
                    else:
                        print("Warning: Could not find JSON in the response")
                except json.JSONDecodeError:
                    print("Warning: Received invalid JSON from Gemini, returning raw response")

            # Process the output based on the selected format
            if textoutput.strip():
                # Clean up the text output
                clean_text = textoutput.strip()

                # Remove any markdown code blocks if present
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        # Extract content between the first set of backticks
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            # Skip the language identifier line
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()

                # Remove any quotes around the text
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()

                # Remove any "Prompt:" or similar prefixes
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break

                # Format as JSON if requested
                if output_format == "json":
                    try:
                        # Create a JSON object with the appropriate key based on the prompt structure
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"

                        json_output = json.dumps({
                            key_name: clean_text
                        }, indent=2)

                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
                    # Just return the clean text
                    textoutput = clean_text
                    print("Returning raw text output")

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
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "SDXL"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Ollama"

    def generate_content(self, prompt, ollama_model, keep_alive, structure_output, prompt_structure, structure_format, output_format, image=None):
        url = f"{self.ollama_url}/api/generate"

        # Apply prompt template
        modified_prompt = apply_prompt_template(prompt, prompt_structure)

        # Add structure format if requested
        if structure_output:
            print(f"Requesting structured output from {ollama_model}")
            # Add the structure format to the prompt
            modified_prompt = f"{modified_prompt}\n\n{structure_format}"
            print(f"Modified prompt with structure format")

        payload = {
            "model": ollama_model,
            "prompt": modified_prompt,
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

            # Get the response text
            textoutput = response.json().get('response', '')

            # Process the output based on the selected format
            if textoutput.strip():
                # Clean up the text output
                clean_text = textoutput.strip()

                # Remove any markdown code blocks if present
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        # Extract content between the first set of backticks
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            # Skip the language identifier line
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()

                # Remove any quotes around the text
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()

                # Remove any "Prompt:" or similar prefixes
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break

                # Format as JSON if requested
                if output_format == "json":
                    try:
                        # Create a JSON object with the appropriate key based on the prompt structure
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"

                        json_output = json.dumps({
                            key_name: clean_text
                        }, indent=2)

                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
                    # Just return the clean text
                    textoutput = clean_text
                    print("Returning raw text output")

            return (textoutput,)
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
    "BRIA_RMBG": "BRIA RMBG",
    "ConvertRasterToVector": "Raster to Vector (SVG)",
    "SaveSVG": "Save SVG",
    "FLUXResolutions": "FLUX Resolutions",
    'ComfyUIStyler': 'ComfyUI Styler'
}
