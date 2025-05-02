import os
import base64
import mimetypes
import torch
import numpy as np
from PIL import Image
import io
from google import genai
from google.genai import types
import folder_paths
import json
from .list_models import get_gemini_image_models

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute cartoon animal in a forest landscape"}),
                "model": (get_gemini_image_models(), {"default": "gemini-2.0-flash-exp-image-generation"}),
                "file_prefix": ("STRING", {"default": "gemini_image"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
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
                import json
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

    def generate_image(self, prompt, model, file_prefix, negative_prompt=None, image=None, image2=None, **kwargs):
        """Handle image generation with flexible inputs - text only, single image, or multiple images"""
        api_key = self.get_gemini_api_key()
        if not api_key:
            return (self.create_empty_image(), "Error: Gemini API key is missing or invalid")
        
        try:
            # Initialize the client with API key
            client = genai.Client(api_key=api_key)
            
            # Keep prompt simple - this works better with the image generation model
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f" (without {negative_prompt})"
            
            # Add a specific request for a colored image (more explicit now)
            full_prompt = f"{full_prompt} (colorful image, vibrant colors, not black and white, not grayscale)" 
                
            # Configure the model request based on model selection
            # model is now passed as a parameter from the UI
            
            # Prepare content parts
            parts = []
            
            # Track if we have any images
            has_images = False
            
            # If we have an input image, save it and add to the request
            if image is not None:
                print("Input image provided, using it for image generation")
                input_image_path = self.save_input_image(image, f"{file_prefix}_1")
                
                # Create a file object for the API
                uploaded_file = client.files.upload(file=input_image_path)
                
                # Add the image as the first part
                parts.append(
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    )
                )
                has_images = True
            
            # If we have a second input image, save it and add to the request
            if image2 is not None:
                print("Second input image provided, adding to image generation")
                input_image_path = self.save_input_image(image2, f"{file_prefix}_2") 
                
                # Create a file object for the API
                uploaded_file = client.files.upload(file=input_image_path)
                
                # Add the image as another part
                parts.append(
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    )
                )
                has_images = True
                
            # Handle any additional images from kwargs (for future flexibility)
            img_index = 3
            for key, value in kwargs.items():
                if key.startswith('image') and isinstance(value, torch.Tensor):
                    print(f"Additional input image {key} provided, adding to image generation")
                    input_image_path = self.save_input_image(value, f"{file_prefix}_{img_index}")
                    
                    # Create a file object for the API
                    uploaded_file = client.files.upload(file=input_image_path)
                    
                    # Add the image as another part
                    parts.append(
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        )
                    )
                    has_images = True
                    img_index += 1
                
            # Add the prompt as text part
            parts.append(types.Part.from_text(text=full_prompt))
            
            if has_images:
                print(f"Including {img_index-1} input images in request with prompt: {full_prompt}")
            else:
                print(f"No input images, using text prompt only: {full_prompt}")
            
            # Create content with the parts
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["image", "text"],
                response_mime_type="text/plain",
            )
            
            # Variables to store results
            image_tensor = None
            response_text = ""
            
            print(f"Generating image with {model} for prompt: {full_prompt}")
            
            # Stream the response and process chunks
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                # Skip invalid chunks
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                    
                # Check for image data
                if chunk.candidates[0].content.parts[0].inline_data:
                    print("Found image data in response")
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    mime_type = inline_data.mime_type
                    
                    # Get the raw image data - don't attempt to decode base64
                    image_data = inline_data.data
                    
                    # Save the raw image data directly using the same method as working example
                    file_extension = mimetypes.guess_extension(mime_type) or ".png"
                    full_output_path = os.path.join(folder_paths.get_output_directory(), f"{file_prefix}{file_extension}")
                    
                    # Use same save function as the working example
                    with open(full_output_path, 'wb') as f:
                        f.write(image_data)
                    
                    print(f"Saved image with mime type {mime_type} to {full_output_path}")
                    
                    # Load directly from the saved file to ensure proper format
                    try:
                        pil_image = Image.open(full_output_path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                    
                        # Convert the PIL image to tensor - in ComfyUI format [B,H,W,C]
                        image = np.array(pil_image).astype(np.float32) / 255.0
                        # Make sure tensor has shape [batch, height, width, channels]
                        image_tensor = torch.from_numpy(image)[None,]
                        print(f"Loaded image: size={pil_image.size}, mode={pil_image.mode}")
                        print(f"Created tensor with shape {image_tensor.shape}")
                        
                        # Debug image center pixel to check color
                        center_x, center_y = pil_image.width // 2, pil_image.height // 2
                        center_pixel = pil_image.getpixel((center_x, center_y))
                        print(f"Center pixel RGB value: {center_pixel}")
                        
                        # Force save colored version for visual verification
                        colored_path = os.path.join(folder_paths.get_output_directory(), f"{file_prefix}_colored.png")
                        pil_image.save(colored_path)
                        print(f"Saved verified colored version to {colored_path}")
                    except Exception as e:
                        print(f"Error loading saved image: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Create a proper RGB black image tensor
                        return (self.create_empty_image(), f"Error loading image: {str(e)}")
                        
                # Check for text data
                elif hasattr(chunk, 'text') and chunk.text:
                    response_text += chunk.text
                    print(f"Text response: {chunk.text}")
            
            # If no image was generated or there was an error processing it
            if image_tensor is None:
                # Create a proper RGB black image tensor
                image_tensor = self.create_empty_image()
                if not response_text:
                    response_text = "No image or text was generated. Try a different prompt."
            
            # No need to normalize or convert - the tensor is already correctly formatted
            # Just make sure we have the expected dimensions [batch_size, height, width, channels]
            print(f"Final tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}, min={image_tensor.min().item()}, max={image_tensor.max().item()}")
            
            return (image_tensor, response_text)
            
        except Exception as e:
            error_message = f"Error generating image with {model}: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            # Create a proper RGB black image tensor
            return (self.create_empty_image(), error_message)

# Register this node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "Gemini Image Generator"
} 