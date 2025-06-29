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
from .clipsegx import GeminiCLIPSeg, GeminiCombineSegMasks
from .BRIA_RMBGx import GeminiBRIA_RMBG
from .svgnodex import GeminiConvertRasterToVector, GeminiSaveSVG
from .FLUXResolutions import GeminiFLUXResolutions
from .prompt_stylerx import NODES

# Try to import torchaudio for audio processing
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("torchaudio not available. Audio processing will be limited.")

# Note for users about audio processing dependencies
"""
For full audio processing support, especially for MP3 files, install these dependencies:
1. torchaudio - Basic audio processing: pip install torchaudio
2. ffmpeg - For converting audio formats: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)
3. pydub - Alternative audio processing: pip install pydub

If you encounter MP3 loading issues, make sure you have the necessary codecs:
- Linux: sudo apt-get install libavcodec-extra
- macOS: brew install ffmpeg --with-libvorbis --with-sdl2 --with-theora
"""

# Common function to apply prompt structure templates
def apply_prompt_template(prompt, prompt_structure="Custom"):
    # Define prompt structure templates
    prompt_templates = {
        "VideoGen": "Create a professional cinematic video generation prompt based on my description. Structure your prompt in this precise order: (1) SUBJECT: Define main character(s)/object(s) with specific, vivid details (appearance, expressions, attributes); (2) CONTEXT/SCENE: Establish the detailed environment with atmosphere, time of day, weather, and spatial relationships; (3) ACTION: Describe precise movements and temporal flow using dynamic verbs and sequential language ('first... then...'); (4) CINEMATOGRAPHY: Specify exact camera movements (dolly, pan, tracking), shot types (close-up, medium, wide), lens choice (35mm, telephoto), and professional lighting terminology (Rembrandt, golden hour, backlit); (5) STYLE: Define the visual aesthetic using specific references to film genres, directors, or animation styles. For realistic scenes, emphasize photorealism with natural lighting and physics. For abstract/VFX, include stylistic terms (surreal, psychedelic) and dynamic descriptors (swirling, morphing). For animation, specify the exact style (anime, 3D cartoon, hand-drawn). Craft a single cohesive paragraph that flows naturally while maintaining technical precision. Return ONLY the prompt text itself no more 200 tokens.",

        "FLUX.1-dev": "As an elite text-to-image prompt engineer, craft an exceptional FLUX.1-dev prompt from my description. Create a hyper-detailed, cinematographic paragraph that includes: (1) precise subject characterization with emotional undertones, (2) specific artistic influences from legendary painters/photographers, (3) technical camera specifications (lens, aperture, perspective), (4) sophisticated lighting setup with exact quality and direction, (5) atmospheric elements and depth effects, (6) composition techniques, and (7) post-processing styles. Use language that balances technical precision with artistic vision. Return ONLY the prompt text itself - no explanations or formatting no more 200 tokens.",

        "SDXL": "Create a premium comma-separated tag prompt for SDXL based on my description. Structure the prompt with these elements in order of importance: (1) main subject with precise descriptors, (2) high-impact artistic medium (oil painting, digital art, photography, etc.), (3) specific art movement or style with named influences, (4) professional lighting terminology (rembrandt, cinematic, golden hour, etc.), (5) detailed environment/setting, (6) exact camera specifications (35mm, telephoto, macro, etc.), (7) composition techniques, (8) color palette/mood, and (9) post-processing effects. Use 20-30 tags maximum, prioritizing quality descriptors over quantity. Include 2-3 relevant artist references whose style matches the desired aesthetic. Return ONLY the comma-separated tags without explanations or formatting.",
        "FLUXKontext": "You are a Flux Kontext prompt generator. Transform user descriptions into precise Flux Kontext editing instructions following this structure: (1) action verb with specific target object, (2) exact modification details, (3) style preservation clauses using 'while maintaining', (4) character consistency with specific descriptors (never use pronouns), (5) precise style names for transformations, (6) quoted text for replacements, and (7) step-by-step approach for complex edits. Use clear, direct language with specific visual descriptors. For style changes, include medium characteristics and named art movements. For character edits, preserve facial features, expressions, and poses explicitly. Return ONLY the Flux Kontext instruction without explanations or formatting."
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

# ================== UNIVERSAL MEDIA UTILITIES ==================
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

def sample_video_frames(video_tensor, num_samples=6):
    """Sample frames evenly from video tensor"""
    if len(video_tensor.shape) != 4:
        return None

    total_frames = video_tensor.shape[0]
    if total_frames <= num_samples:
        indices = range(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        frame = tensor_to_pil_image(video_tensor[idx])
        frames.append(frame)
    return frames

def process_audio(audio_data, target_sample_rate=16000):
    """Process audio data for API submission with robust error handling"""
    if not TORCHAUDIO_AVAILABLE:
        print("Warning: torchaudio not available, cannot process audio")
        return None

    try:
        # Check if we received a path or a tensor
        if isinstance(audio_data, str):
            # It's a file path
            try:
                print(f"Loading audio from path: {audio_data}")
                # Try to load with torchaudio
                try:
                    waveform, sample_rate = torchaudio.load(audio_data)
                except RuntimeError as e:
                    print(f"Error loading with torchaudio: {str(e)}")
                    # If MP3 loading fails, try using alternative methods
                    if audio_data.lower().endswith('.mp3'):
                        print("Attempting to load MP3 with alternative method...")
                        try:
                            # Try to use ffmpeg if available
                            import subprocess
                            import tempfile

                            # Create a temporary WAV file
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                                temp_wav_path = temp_wav.name

                            # Convert MP3 to WAV using ffmpeg
                            cmd = ["ffmpeg", "-i", audio_data, "-ar", str(target_sample_rate), "-ac", "1", temp_wav_path]
                            subprocess.run(cmd, check=True, capture_output=True)

                            # Load the WAV file
                            waveform, sample_rate = torchaudio.load(temp_wav_path)

                            # Clean up
                            os.remove(temp_wav_path)
                            print("Successfully converted and loaded MP3 file")
                        except Exception as ffmpeg_error:
                            print(f"Failed to use ffmpeg: {str(ffmpeg_error)}")
                            # If ffmpeg fails, try one more fallback
                            try:
                                from pydub import AudioSegment
                                import numpy as np

                                print("Attempting to load with pydub...")
                                sound = AudioSegment.from_mp3(audio_data)
                                sound = sound.set_frame_rate(target_sample_rate).set_channels(1)
                                samples = np.array(sound.get_array_of_samples())
                                waveform = torch.tensor(samples).float().div_(32768.0).unsqueeze(0)
                                sample_rate = target_sample_rate
                                print("Successfully loaded MP3 with pydub")
                            except Exception as pydub_error:
                                print(f"Failed to use pydub: {str(pydub_error)}")
                                raise RuntimeError("All methods to load MP3 failed")
                    else:
                        # For non-MP3 files, re-raise the original error
                        raise
            except Exception as load_error:
                print(f"All methods to load audio file failed: {str(load_error)}")
                return None
        else:
            # It's a tensor or dictionary
            try:
                waveform = audio_data["waveform"]
                sample_rate = audio_data["sample_rate"]
            except (TypeError, KeyError):
                # If it's just a tensor
                if torch.is_tensor(audio_data):
                    waveform = audio_data
                    sample_rate = target_sample_rate  # Assume target sample rate
                else:
                    print(f"Unsupported audio data format: {type(audio_data)}")
                    return None

        # Handle different dimensions
        if waveform.dim() == 3:  # [batch, channels, time]
            waveform = waveform.squeeze(0)
        elif waveform.dim() == 1:  # [time]
            waveform = waveform.unsqueeze(0)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

        # Normalize audio if needed
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        # Convert to WAV format
        buffer = io.BytesIO()
        try:
            torchaudio.save(buffer, waveform, target_sample_rate, format="WAV")
            audio_bytes = buffer.getvalue()
            return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as save_error:
            print(f"Error saving audio to buffer: {str(save_error)}")
            return None

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None
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

def update_config_key(key, value):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    try:
        # Read existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/corrupt, start with a new dict
        config = {}
    
    # Update the key
    config[key] = value
    
    # Write the updated config back
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Successfully updated {key} in config.json")
    except Exception as e:
        print(f"Error writing to config.json: {str(e)}")

# ================== API SERVICES ==================


class GeminiQwenAPI:
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
                    "SDXL",
                    "FLUXKontext"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Qwen"

    def generate_content(self, prompt, qwen_model, max_tokens, temperature, top_p, structure_output, prompt_structure, structure_format, output_format, api_key="", image=None):
        if api_key:
            update_config_key("QWEN_API_KEY", api_key)
            self.qwen_api_key = api_key
            # Re-initialize the client with the new key
            self.client = OpenAI(
                api_key=self.qwen_api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

        if not self.qwen_api_key:
            return ("Qwen API key missing. Please provide it in the node's api_key input.",)

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

class GeminiOpenAIAPI:
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

        # Import model list from list_models.py
        from .list_models import get_openai_models
        self.available_models = get_openai_models()

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
        # Create an instance to get the models
        instance = cls()
        available_models = instance.available_models

        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "model": (available_models,),
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
                    "SDXL",
                    "FLUXKontext"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "openai_api_key": ("STRING", {"default": "", "multiline": False}),
                "nvidia_api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/OpenAI"

    def generate_content(self, prompt, model, max_tokens, temperature, top_p, stream, structure_output, prompt_structure, structure_format, output_format, openai_api_key="", nvidia_api_key="", image=None):
        if openai_api_key:
            update_config_key("OPENAI_API_KEY", openai_api_key)
            self.openai_api_key = openai_api_key
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        if nvidia_api_key:
            update_config_key("NVIDIA_API_KEY", nvidia_api_key)
            self.nvidia_api_key = nvidia_api_key
            self.nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.nvidia_api_key
            )

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

class GeminiClaudeAPI:
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
                    "SDXL",
                    "FLUXKontext"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Claude"

    def generate_content(self, prompt, model, max_tokens, structure_output, prompt_structure, structure_format, output_format, api_key="", image=None):
        if api_key:
            update_config_key("CLAUDE_API_KEY", api_key)
            self.claude_api_key = api_key
            self.client = anthropic.Client(api_key=self.claude_api_key)

        if not self.claude_api_key:
            return ("Claude API key missing. Please provide it in the node's api_key input.",)

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

class GeminiLLMAPI:
    def __init__(self):
        self.gemini_api_key = get_gemini_api_key()
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key, transport='rest')

        # Import model list from list_models.py
        from .list_models import get_gemini_models
        self.available_models = get_gemini_models()

    @classmethod
    def INPUT_TYPES(cls):
        # Create an instance to get the models
        instance = cls()
        available_models = instance.available_models

        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "gemini_model": (available_models,),
                "stream": ("BOOLEAN", {"default": False}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "VideoGen",
                    "FLUX.1-dev",
                    "SDXL",
                    "FLUXKontext"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "video": ("IMAGE",),  # Video is represented as a tensor with shape [frames, height, width, channels]
                "audio": ("AUDIO",),  # Audio input
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Gemini"

    def generate_content(self, prompt, input_type, gemini_model, stream, structure_output, prompt_structure, structure_format, output_format, api_key="", image=None, video=None, audio=None):
        if api_key:
            update_config_key("GEMINI_API_KEY", api_key)
            self.gemini_api_key = api_key
            genai.configure(api_key=self.gemini_api_key, transport='rest')

        if not self.gemini_api_key:
            return ("Gemini API key missing. Please provide it in the node's api_key input.",)

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }

            # Apply prompt template
            modified_prompt = apply_prompt_template(prompt, prompt_structure)

            # Add JSON structure format if requested
            if structure_output:
                print(f"Requesting structured output from {gemini_model}")
                # Add the structure format to the prompt
                modified_prompt = f"{modified_prompt}\n\n{structure_format}"
                print(f"Modified prompt with structure format")

            # Create the model
            model = genai.GenerativeModel(gemini_model)

            # Process different input types
            if input_type == "text":
                # Text-only input
                print(f"Processing text input for Gemini API")
                content = [modified_prompt]

            elif input_type == "image" and image is not None:
                # Process image input
                print(f"Processing image input for Gemini API")
                pil_image = tensor_to_pil_image(image)
                content = [modified_prompt, pil_image]

            elif input_type == "video" and video is not None:
                # Process video input (extract frames)
                print(f"Processing video input for Gemini API")
                frames = sample_video_frames(video)
                if frames:
                    # Create content with text and frames
                    content = [modified_prompt]
                    for frame in frames:
                        content.append(frame)

                    # Update prompt to indicate video analysis
                    frame_count = len(frames)
                    modified_prompt = f"Analyze these {frame_count} frames from a video: {modified_prompt}"
                    content[0] = modified_prompt
                else:
                    print("Error: Could not extract frames from video")
                    return ("Error: Could not extract frames from video",)

            elif input_type == "audio" and audio is not None:
                # Process audio input
                print(f"Processing audio input for Gemini API")
                if not TORCHAUDIO_AVAILABLE:
                    return ("Error: torchaudio not available for audio processing",)

                try:
                    # Check different audio input formats
                    if isinstance(audio, dict):
                        if "path" in audio:
                            # Direct path format
                            audio_path = audio["path"]
                            print(f"Processing audio from path: {audio_path}")
                            audio_b64 = process_audio(audio_path)
                        elif "waveform" in audio and "sample_rate" in audio:
                            # ComfyUI audio node format
                            print(f"Processing audio from waveform tensor")
                            audio_b64 = process_audio(audio)
                        else:
                            # Unknown dictionary format
                            print(f"Unknown audio dictionary format: {list(audio.keys())}")
                            return ("Error: Unsupported audio format",)
                    elif isinstance(audio, str) and os.path.exists(audio):
                        # Direct file path
                        print(f"Processing audio from direct path: {audio}")
                        audio_b64 = process_audio(audio)
                    else:
                        # Try to process as tensor or other format
                        print(f"Attempting to process audio as tensor")
                        audio_b64 = process_audio(audio)

                    if audio_b64:
                        # Gemini doesn't directly support audio in the Python SDK
                        # We'll use the REST API directly for audio
                        try:
                            import requests

                            # Prepare the request
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
                            headers = {
                                "Content-Type": "application/json",
                                "x-goog-api-key": self.gemini_api_key
                            }

                            # Create the request body
                            request_body = {
                                "contents": [{
                                    "parts": [
                                        {"text": modified_prompt},
                                        {
                                            "inline_data": {
                                                "mime_type": "audio/wav",
                                                "data": audio_b64
                                            }
                                        }
                                    ]
                                }],
                                "generation_config": generation_config,
                                "safety_settings": [
                                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                                ]
                            }

                            # Send the request
                            print(f"Sending audio request to Gemini API REST endpoint")
                            response = requests.post(url, json=request_body, headers=headers)
                            response.raise_for_status()

                            # Extract the response text
                            response_json = response.json()
                            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                                candidate = response_json["candidates"][0]
                                if "content" in candidate and "parts" in candidate["content"]:
                                    parts = candidate["content"]["parts"]
                                    text_parts = [part.get("text", "") for part in parts if "text" in part]
                                    textoutput = " ".join(text_parts)
                                    return (textoutput,)

                            return ("Error: Could not parse Gemini API response for audio",)
                        except Exception as e:
                            print(f"Error using REST API for audio: {str(e)}")
                            # Fallback to text-only with a note about audio
                            content = [f"[This prompt was supposed to include audio data, but audio processing failed: {str(e)}] {modified_prompt}"]
                    else:
                        return ("Error: Failed to process audio data",)
                except Exception as e:
                    print(f"Error processing audio for Gemini: {str(e)}")
                    return (f"Error processing audio: {str(e)}",)
            else:
                # Default to text-only
                content = [modified_prompt]

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

class GeminiOllamaAPI:
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
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "ollama_model": (cls.get_ollama_models(),),
                "keep_alive": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "VideoGen",
                    "FLUX.1-dev",
                    "SDXL",
                    "FLUXKontext"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),  # Video is represented as a tensor with shape [frames, height, width, channels]
                "audio": ("AUDIO",),  # Audio input
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Ollama"

    def generate_content(self, prompt, input_type, ollama_model, keep_alive, structure_output, prompt_structure, structure_format, output_format, image=None, video=None, audio=None):
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
            # Process different input types
            if input_type == "text":
                # Text-only input, no additional processing needed
                print(f"Processing text input for Ollama API")

            elif input_type == "image" and image is not None:
                # Process image input
                print(f"Processing image input for Ollama API")
                pil_image = tensor_to_pil_image(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                payload["images"] = [base64.b64encode(buffered.getvalue()).decode()]

                # Update prompt to indicate image analysis
                modified_prompt = f"Analyze this image: {modified_prompt}"
                payload["prompt"] = modified_prompt

            elif input_type == "video" and video is not None:
                # Process video input (extract frames)
                print(f"Processing video input for Ollama API")
                frames = sample_video_frames(video)
                if frames:
                    # Convert frames to base64
                    frame_data = []
                    for frame in frames:
                        buffered = io.BytesIO()
                        frame.save(buffered, format="PNG")
                        frame_data.append(base64.b64encode(buffered.getvalue()).decode())

                    # Add frames to payload
                    payload["images"] = frame_data

                    # Update prompt to indicate video analysis
                    frame_count = len(frames)
                    modified_prompt = f"Analyze these {frame_count} frames from a video: {modified_prompt}"
                    payload["prompt"] = modified_prompt
                else:
                    print("Error: Could not extract frames from video")
                    return ("Error: Could not extract frames from video",)

            elif input_type == "audio" and audio is not None:
                # Process audio input
                print(f"Processing audio input for Ollama API")
                if not TORCHAUDIO_AVAILABLE:
                    return ("Error: torchaudio not available for audio processing",)

                try:
                    # Check different audio input formats
                    if isinstance(audio, dict):
                        if "path" in audio:
                            # Direct path format
                            audio_path = audio["path"]
                            print(f"Processing audio from path: {audio_path}")
                            audio_b64 = process_audio(audio_path)
                        elif "waveform" in audio and "sample_rate" in audio:
                            # ComfyUI audio node format
                            print(f"Processing audio from waveform tensor")
                            audio_b64 = process_audio(audio)
                        else:
                            # Unknown dictionary format
                            print(f"Unknown audio dictionary format: {list(audio.keys())}")
                            return ("Error: Unsupported audio format",)
                    elif isinstance(audio, str) and os.path.exists(audio):
                        # Direct file path
                        print(f"Processing audio from direct path: {audio}")
                        audio_b64 = process_audio(audio)
                    else:
                        # Try to process as tensor or other format
                        print(f"Attempting to process audio as tensor")
                        audio_b64 = process_audio(audio)

                    if audio_b64:
                        # Ollama doesn't directly support audio, so we'll include a note in the prompt
                        modified_prompt = f"[This prompt includes audio data that has been processed] {modified_prompt}"
                        payload["prompt"] = modified_prompt

                        # Some Ollama models might support base64 encoded audio as an image
                        # This is experimental and may not work with all models
                        payload["images"] = [audio_b64]
                    else:
                        return ("Error: Failed to process audio data",)
                except Exception as e:
                    print(f"Error processing audio for Ollama: {str(e)}")
                    return (f"Error processing audio: {str(e)}",)

            # Send request to Ollama API
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
class GeminiTextSplitByDelimiter:
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

class GeminiSaveTextFile:
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

# Add a node to display available models
class GeminiListAvailableModels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "display_gemini": ("BOOLEAN", {"default": True}),
                "display_openai": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "list_models"
    CATEGORY = "AI API/Utils"

    def list_models(self, display_gemini, display_openai):
        from .list_models import get_gemini_models, get_openai_models
        model_list = []

        if display_gemini:
            gemini_models = get_gemini_models()
            model_list.append("=== Gemini Models ===")
            model_list.extend(gemini_models)
            model_list.append("")

        if display_openai:
            openai_models = get_openai_models()
            model_list.append("=== OpenAI Models ===")
            model_list.extend(openai_models)

        return ("\n".join(model_list),)

# ================== NODE REGISTRATION ==================
NODE_CLASS_MAPPINGS = {
    "GeminiAPI": GeminiLLMAPI,
    "OllamaAPI": GeminiOllamaAPI,
    "OpenAIAPI": GeminiOpenAIAPI,
    "ClaudeAPI": GeminiClaudeAPI,
    "QwenAPI": GeminiQwenAPI,
    "GeminiTextSplitter": GeminiTextSplitByDelimiter,
    "GeminiSaveText": GeminiSaveTextFile,
    "ListAvailableModels": GeminiListAvailableModels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAPI": "Gemini API",
    "OllamaAPI": "Ollama API",
    "OpenAIAPI": "OpenAI API",
    "ClaudeAPI": "Claude API",
    "QwenAPI": "Qwen API",
    "GeminiTextSplitter": "Gemini Text Splitter",
    "GeminiSaveText": "Gemini Save Text",
    "ListAvailableModels": "List Available Models",
}
