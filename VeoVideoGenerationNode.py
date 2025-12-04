"""
Veo 3.1 Video Generation Node for ComfyUI
Supports: Text-to-Video, Image-to-Video, Video Extension, Reference Images, First/Last Frame Interpolation
"""

import os
import torch
import numpy as np
from PIL import Image
import io
import time
import google.genai as genai
from google.genai import types
import folder_paths
import json

def tensor_to_pil_image(tensor):
    """Convert tensor to PIL Image"""
    if tensor is None:
        return None
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    
    channels = image_np.shape[-1]
    mode = 'RGBA' if channels == 4 else 'RGB'
    image = Image.fromarray(image_np, mode=mode)
    
    # Convert RGBA to RGB
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    return image


class VeoVideoGenerator:
    """
    Generate videos using Google's Veo 3.1 model.
    
    Features:
    - Text-to-Video: Generate 8-second 720p/1080p videos from text prompts
    - Image-to-Video: Use an image as the first frame
    - First/Last Frame: Specify both start and end frames for interpolation
    - Reference Images: Use up to 3 reference images to guide content
    - Video Extension: Extend previously generated Veo videos
    - Native Audio: Veo 3.1 generates synchronized audio with dialogue & SFX
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A cinematic shot of a majestic lion walking through the golden savannah at sunset, warm lighting, professional wildlife documentary style."
                }),
                "model": ([
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                    "veo-3-generate-preview",
                    "veo-3-fast-generate-preview",
                    "veo-2.0-generate-001"
                ], {"default": "veo-3.1-generate-preview"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration_seconds": (["4", "6", "8"], {"default": "8"}),
                "person_generation": (["allow_all", "allow_adult", "dont_allow"], {"default": "allow_all"}),
                "file_prefix": ("STRING", {"default": "veo_video"}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                # Image inputs
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                # For video extension
                "extend_video_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "status", "operation_name")
    FUNCTION = "generate_video"
    CATEGORY = "Gemini/Video"
    
    def get_gemini_api_key(self):
        """Get Gemini API key from config or environment"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get("GEMINI_API_KEY", "")
                if api_key and api_key != "your_gemini_api_key_here":
                    return api_key
        return os.environ.get("GEMINI_API_KEY", "")

    def generate_video(self, prompt, model, aspect_ratio, resolution, duration_seconds,
                       person_generation, file_prefix, poll_interval,
                       negative_prompt=None, seed=-1,
                       first_frame=None, last_frame=None,
                       reference_image_1=None, reference_image_2=None, reference_image_3=None,
                       extend_video_path=None, **kwargs):
        
        api_key = self.get_gemini_api_key()
        if not api_key:
            return ("", "Error: Gemini API key is missing. Please set it in config.json or GEMINI_API_KEY environment variable.", "")
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Build generation config
            config_params = {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration_seconds": int(duration_seconds),
                "person_generation": person_generation,
                "number_of_videos": 1,
            }
            
            # Add negative prompt if provided
            if negative_prompt and negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()
            
            # Add seed if provided (not -1)
            if seed != -1:
                config_params["seed"] = seed
            
            # Determine generation mode
            generation_mode = "text-to-video"
            operation = None
            
            # MODE 1: Video Extension (Veo 3.1 only)
            if extend_video_path and extend_video_path.strip() and os.path.exists(extend_video_path):
                if "3.1" not in model:
                    return ("", "Error: Video extension is only supported with Veo 3.1 models.", "")
                
                generation_mode = "video-extension"
                print(f"🎬 Veo Video Extension mode: {extend_video_path}")
                
                # Read video file
                with open(extend_video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Upload video to API
                video_file = client.files.upload(
                    file=io.BytesIO(video_bytes),
                    config={"mime_type": "video/mp4"}
                )
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    video=video_file,
                    config=types.GenerateVideosConfig(**config_params)
                )
            
            # MODE 2: First/Last Frame Interpolation (Veo 3.1 only)
            elif first_frame is not None and last_frame is not None:
                if "3.1" not in model:
                    return ("", "Error: First/Last frame interpolation is only supported with Veo 3.1 models.", "")
                
                generation_mode = "interpolation"
                print(f"🎬 Veo Interpolation mode (first + last frame)")
                
                first_pil = tensor_to_pil_image(first_frame)
                last_pil = tensor_to_pil_image(last_frame)
                
                config_params["last_frame"] = last_pil
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=first_pil,
                    config=types.GenerateVideosConfig(**config_params)
                )
            
            # MODE 3: Reference Images (Veo 3.1 only)
            elif any([reference_image_1 is not None, reference_image_2 is not None, reference_image_3 is not None]):
                if "3.1" not in model:
                    return ("", "Error: Reference images are only supported with Veo 3.1 models.", "")
                
                generation_mode = "reference-images"
                reference_images = []
                
                for idx, ref_img in enumerate([reference_image_1, reference_image_2, reference_image_3], 1):
                    if ref_img is not None:
                        pil_img = tensor_to_pil_image(ref_img)
                        ref_obj = types.VideoGenerationReferenceImage(
                            image=pil_img,
                            reference_type="asset"
                        )
                        reference_images.append(ref_obj)
                        print(f"📷 Added reference image {idx}")
                
                print(f"🎬 Veo Reference Images mode: {len(reference_images)} images")
                
                config_params["reference_images"] = reference_images
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(**config_params)
                )
            
            # MODE 4: Image-to-Video (single first frame)
            elif first_frame is not None:
                generation_mode = "image-to-video"
                print(f"🎬 Veo Image-to-Video mode")
                
                first_pil = tensor_to_pil_image(first_frame)
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=first_pil,
                    config=types.GenerateVideosConfig(**config_params)
                )
            
            # MODE 5: Text-to-Video (default)
            else:
                generation_mode = "text-to-video"
                print(f"🎬 Veo Text-to-Video mode")
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(**config_params)
                )
            
            print(f"📤 Started Veo generation: {generation_mode}")
            print(f"   Model: {model}")
            print(f"   Aspect Ratio: {aspect_ratio}, Resolution: {resolution}")
            print(f"   Duration: {duration_seconds}s")
            print(f"   Operation: {operation.name}")
            
            # Poll for completion
            poll_count = 0
            max_polls = 60  # Max 10 minutes at 10s intervals
            
            while not operation.done and poll_count < max_polls:
                poll_count += 1
                print(f"⏳ Waiting for video generation... ({poll_count * poll_interval}s elapsed)")
                time.sleep(poll_interval)
                operation = client.operations.get(operation)
            
            if not operation.done:
                return ("", f"Timeout: Video generation did not complete within {max_polls * poll_interval} seconds. Operation: {operation.name}", operation.name)
            
            # Check for errors
            if operation.error:
                return ("", f"Error: {operation.error}", operation.name)
            
            # Download the generated video
            if not operation.response or not operation.response.generated_videos:
                return ("", "Error: No video was generated.", operation.name)
            
            generated_video = operation.response.generated_videos[0]
            
            # Download video
            client.files.download(file=generated_video.video)
            
            # Save to output directory
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            video_filename = f"{file_prefix}_{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            generated_video.video.save(video_path)
            
            print(f"✅ Video saved to: {video_path}")
            
            status_msg = f"Video generated successfully!\nMode: {generation_mode}\nModel: {model}\nDuration: {duration_seconds}s\nResolution: {resolution}\nSaved to: {video_filename}"
            
            return (video_path, status_msg, operation.name)
            
        except Exception as e:
            error_msg = f"Error generating video: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", error_msg, "")


class VeoVideoExtender:
    """
    Extend a previously generated Veo video by up to 7 seconds.
    Can extend videos up to 20 times (max 148 seconds total).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Continue the scene with smooth motion and natural progression."
                }),
                "file_prefix": ("STRING", {"default": "veo_extended"}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "status")
    FUNCTION = "extend_video"
    CATEGORY = "Gemini/Video"
    
    def get_gemini_api_key(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get("GEMINI_API_KEY", "")
                if api_key and api_key != "your_gemini_api_key_here":
                    return api_key
        return os.environ.get("GEMINI_API_KEY", "")
    
    def extend_video(self, video_path, prompt, file_prefix, poll_interval, negative_prompt=None, **kwargs):
        api_key = self.get_gemini_api_key()
        if not api_key:
            return ("", "Error: Gemini API key is missing.")
        
        if not video_path or not os.path.exists(video_path):
            return ("", f"Error: Video file not found: {video_path}")
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Read and upload video
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            video_file = client.files.upload(
                file=io.BytesIO(video_bytes),
                config={"mime_type": "video/mp4"}
            )
            
            config_params = {
                "number_of_videos": 1,
                "resolution": "720p",
            }
            
            if negative_prompt and negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()
            
            print(f"🎬 Extending video: {video_path}")
            
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                video=video_file,
                config=types.GenerateVideosConfig(**config_params)
            )
            
            # Poll for completion
            poll_count = 0
            max_polls = 60
            
            while not operation.done and poll_count < max_polls:
                poll_count += 1
                print(f"⏳ Extending video... ({poll_count * poll_interval}s elapsed)")
                time.sleep(poll_interval)
                operation = client.operations.get(operation)
            
            if not operation.done:
                return ("", f"Timeout: Video extension did not complete.")
            
            if operation.error:
                return ("", f"Error: {operation.error}")
            
            if not operation.response or not operation.response.generated_videos:
                return ("", "Error: No extended video was generated.")
            
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            video_filename = f"{file_prefix}_{timestamp}.mp4"
            output_path = os.path.join(output_dir, video_filename)
            
            generated_video.video.save(output_path)
            
            print(f"✅ Extended video saved to: {output_path}")
            
            return (output_path, f"Video extended successfully! Saved to: {video_filename}")
            
        except Exception as e:
            error_msg = f"Error extending video: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", error_msg)


class VeoPromptTemplates:
    """
    Pre-built prompt templates optimized for Veo 3.1 video generation.
    Based on Google's official Veo prompt guide.
    """
    
    TEMPLATES = {
        "Cinematic-Dialogue": '''A close up of two people staring at {scene_description}, torchlight flickering.
{character_1} murmurs, "{dialogue_1}" {character_2} looks at them and whispering excitedly, "{dialogue_2}"''',
        
        "Cinematic-Realism": '''A cinematic {shot_type} shot of {subject} in {environment}. 
{action_description}. 
The scene features {lighting_description} lighting, creating a {mood} atmosphere. 
Camera: {camera_motion}. Style: {style_keywords}.''',
        
        "Creative-Animation": '''{animation_style} animation. {subject} {action} in {environment}. 
{character_dialogue}
Style: {visual_style}, {color_palette} colors.''',
        
        "Wildlife-Documentary": '''A {shot_type} shot of {animal} {action} in {habitat}. 
{time_of_day} lighting with {weather_conditions}. 
Professional wildlife documentary cinematography, National Geographic quality.
{ambient_sounds}''',
        
        "Product-Commercial": '''A sleek, professional commercial shot of {product}. 
{camera_motion} revealing {product_features}. 
Studio lighting with {lighting_style}. 
Clean, modern aesthetic suitable for {brand_type} advertising.''',
        
        "Music-Video": '''A {style} music video scene. {performer} {action} in {setting}. 
Dynamic {camera_movement} with {visual_effects}. 
{lighting_mood} lighting, {color_grading} color grade.
Beat-synchronized editing style.''',
        
        "Nature-Timelapse": '''A stunning timelapse of {nature_subject} {transformation}. 
{time_span} compressed into seconds. 
{weather_changes}. Smooth motion, professional cinematography.
{ambient_audio}''',
        
        "Horror-Suspense": '''A {shot_type} horror scene. {subject} {action} in {creepy_location}. 
{eerie_element}. The atmosphere is tense and foreboding.
{sound_design}
Dark, desaturated color palette with {accent_color} highlights.''',
        
        "Sci-Fi-Fantasy": '''A {shot_type} shot in a {setting_type} world. {subject} {action}. 
{visual_elements}. 
{lighting_description} with {special_effects}.
Epic, cinematic scale. {audio_elements}''',

        "Social-Media-Short": '''A vertical video for {platform}. {hook_action} in the first second.
{main_content}
Engaging, fast-paced editing. {trending_style}.
{call_to_action}'''
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (list(cls.TEMPLATES.keys()), {"default": "Cinematic-Realism"}),
                "custom_values": ("STRING", {
                    "multiline": True,
                    "default": '''subject: a majestic eagle
environment: snow-capped mountains at dawn
action_description: soaring gracefully through the clouds
shot_type: wide aerial
lighting_description: golden hour
mood: awe-inspiring
camera_motion: slow tracking shot following the eagle
style_keywords: IMAX quality, 8K resolution'''
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Gemini/Video"
    
    def generate_prompt(self, template, custom_values):
        template_text = self.TEMPLATES.get(template, "")
        
        # Parse custom values
        values = {}
        for line in custom_values.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                values[key.strip()] = value.strip()
        
        # Replace placeholders
        result = template_text
        for key, value in values.items():
            result = result.replace(f'{{{key}}}', value)
        
        # Remove any unfilled placeholders
        import re
        result = re.sub(r'\{[^}]+\}', '', result)
        
        return (result.strip(),)


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "VeoVideoGenerator": VeoVideoGenerator,
    "VeoVideoExtender": VeoVideoExtender,
    "VeoPromptTemplates": VeoPromptTemplates,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoVideoGenerator": "Veo 3.1 Video Generator",
    "VeoVideoExtender": "Veo Video Extender",
    "VeoPromptTemplates": "Veo Prompt Templates",
}
