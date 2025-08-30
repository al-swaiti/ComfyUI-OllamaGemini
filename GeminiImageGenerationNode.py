import os
import torch
import numpy as np
from PIL import Image
import io
import google.genai as genai
from google.genai import types
import folder_paths
import json
from .list_models import get_gemini_image_models

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

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        image_models = get_gemini_image_models()
        # Set the new preview model as the default if it's available.
        default_model = "gemini-2.5-flash-image-preview" if "gemini-2.5-flash-image-preview" in image_models else image_models[0] if image_models else ""

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute cartoon animal in a forest landscape"}),
                "model": (image_models, {"default": default_model}),
                "aspect_ratio": (["1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "1:1"}),
                "image_size": (["1K", "2K"], {"default": "1K"}),
                "file_prefix": ("STRING", {"default": "gemini_image"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "AI API/Gemini"

    DESCRIPTION = """Generate images using Google's AI models for image generation:

- gemini-2.0-flash-exp-image-generation: Experimental Gemini model for image generation.
  Fast model that can generate images from text prompts with good accuracy.

- imagen-3.0-generate-002: Google's Imagen 3.0 model for high-quality image generation.
  A more advanced model for detailed, high-fidelity image generation with stronger prompt following.

Both models support reference images to guide the generation process.
When using reference images, the model will try to generate content that's visually similar
or stylistically aligned with the provided images.

Note: These models require proper API access and may have usage limitations or costs associated
with them. Check your Google Cloud/Vertex AI account for details.
    """

    def get_gemini_api_key(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get("GEMINI_API_KEY", "")
            if not api_key:
                return None
            return api_key
        except Exception as e:
            print(f"Error loading Gemini API key: {str(e)}")
            return None

    def save_input_image(self, image_tensor, filename_prefix):
        """Save the input tensor as an image file"""
        # Convert from tensor [B,H,W,C] to PIL Image
        if image_tensor.shape[0] > 1:
            print(f"Warning: Input has {image_tensor.shape[0]} images, using only the first one")

        # Extract first image if it's a batch
        image = image_tensor[0].cpu().numpy()

        # Ensure values are in 0-1 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Save to output directory
        output_dir = folder_paths.get_output_directory()
        temp_image_path = os.path.join(output_dir, f"{filename_prefix}_input.png")
        pil_image.save(temp_image_path)

        print(f"Saved input image to {temp_image_path}")
        return temp_image_path

    def create_empty_image(self, height=512, width=512):
        """Create an empty black image tensor with proper format for ComfyUI"""
        # Create black RGB image with shape [batch, height, width, channels]
        return torch.zeros((1, height, width, 3), dtype=torch.float32)

    def generate_image(self, prompt, model, aspect_ratio, image_size, file_prefix, negative_prompt=None, image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, **kwargs):
        api_key = self.get_gemini_api_key()
        if not api_key:
            return (self.create_empty_image(), "Error: Gemini API key is missing or invalid")

        try:
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f" (Negative prompt: {negative_prompt})"

            client = genai.Client(api_key=api_key)

            # --- Smart API Method Selection ---

            # Case 1: Use the 'generate_images' method for Imagen models.
            if 'imagen' in model:
                print(f"Using 'generate_images' API for Imagen model: {model}")

                # Import the types module to create the config object
                from google.genai import types

                config = types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                )

                response = client.models.generate_images(
                    model=model,
                    prompt=full_prompt,
                    config=config,
                )

                if not response.generated_images:
                    return (self.create_empty_image(), "Error: The API did not return any images.")

                image_bytes = response.generated_images[0].image.image_bytes

            # Case 2: Use the 'generate_content' method via the client for Gemini models.
            else:
                print(f"Using 'generate_content' API for Gemini model: {model}")

                all_images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
                provided_images = [img for img in all_images if img is not None]

                contents = [full_prompt]
                if provided_images:
                    for img in provided_images:
                        contents.append(tensor_to_pil_image(img))

                response = client.models.generate_content(
                    model=model,
                    contents=contents
                )

                # Add robust checks for the response object
                if not response.candidates:
                    try:
                        error_info = response.prompt_feedback
                        return (self.create_empty_image(), f"Error: Prompt was blocked. Feedback: {error_info}")
                    except (AttributeError, IndexError):
                        return (self.create_empty_image(), "Error: The API did not return any candidates.")

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    return (self.create_empty_image(), "Error: The API response did not contain any image data.")

                image_parts = [part for part in candidate.content.parts if part.inline_data]
                if not image_parts:
                    text_parts = [part.text for part in candidate.content.parts if part.text]
                    if text_parts:
                        return (self.create_empty_image(), f"API Error: {text_parts[0]}")
                    return (self.create_empty_image(), "Error: The API response did not contain an image.")

                image_bytes = image_parts[0].inline_data.data

            # --- Process the resulting image bytes ---
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            response_text = "Image generated successfully."
            print(f"Final tensor shape: {image_tensor.shape}")
            return (image_tensor, response_text)

        except Exception as e:
            error_message = f"Error generating image with {model}: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (self.create_empty_image(), error_message)

# Register this node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "Gemini Image Generator"
}
