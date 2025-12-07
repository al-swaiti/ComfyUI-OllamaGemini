"""
Veo 3.1 Video Generation Node for ComfyUI
Based on official Google Veo API: https://ai.google.dev/gemini-api/docs/video

Features:
- Text-to-video generation
- Image-to-video (first frame)
- Reference images (up to 3) for Veo 3.1
- First + Last frame interpolation for Veo 3.1
- Seed support for Veo 3 models
"""

import os
import torch
import numpy as np
from PIL import Image
import time
import uuid
import io
from google import genai
from google.genai import types
import folder_paths
import json


def tensor_to_pil_image(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    if tensor is None:
        return None
    tensor = tensor.cpu()
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch
    image_np = tensor.mul(255).clamp(0, 255).byte().numpy()
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    return Image.fromarray(image_np, 'RGB')


def pil_to_genai_image(pil_image):
    """Convert PIL Image to google.genai types.Image format"""
    if pil_image is None:
        return None
    
    # Convert PIL to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    # Create genai Image with imageBytes and mimeType parameters
    return types.Image(imageBytes=image_bytes, mimeType="image/png")


def tensor_to_genai_image(tensor):
    """Convert ComfyUI tensor directly to genai Image format"""
    pil_image = tensor_to_pil_image(tensor)
    if pil_image is None:
        return None
    return pil_to_genai_image(pil_image)


def extract_frames_and_audio(video_path):
    """Extract frames and audio using PyAV"""
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None, None, 24.0, 0
    
    try:
        import av
        
        frames = []
        audio_data = None
        sample_rate = 44100
        
        with av.open(video_path, mode='r') as container:
            # Get video stream info
            video_stream = container.streams.video[0]
            video_fps = float(video_stream.average_rate) if video_stream.average_rate else 24.0
            
            # Extract video frames
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                img_float = torch.from_numpy(img.astype(np.float32) / 255.0)
                frames.append(img_float)
            
            # Extract audio if present
            container.seek(0)
            for stream in container.streams:
                if stream.type == 'audio':
                    audio_frames = []
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            audio_frames.append(frame.to_ndarray())
                    if audio_frames:
                        audio_np = np.concatenate(audio_frames, axis=1)
                        waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
                        sample_rate = int(stream.sample_rate) if stream.sample_rate else 44100
                        audio_data = {"waveform": waveform, "sample_rate": sample_rate}
                        print(f"✅ Audio extracted: {waveform.shape}, {sample_rate} Hz")
                    break
        
        if not frames:
            print("❌ No frames extracted")
            return None, None, video_fps, 0
        
        frames_tensor = torch.stack(frames)
        print(f"✅ Extracted {len(frames)} frames at {video_fps} FPS")
        return frames_tensor, audio_data, video_fps, len(frames)
        
    except Exception as e:
        print(f"❌ Frame extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 24.0, 0


def create_empty_audio():
    return {"waveform": torch.zeros(1, 2, 44100), "sample_rate": 44100}


class VeoVideoGenerator:
    """
    Generate videos using Google's Veo API.
    Based on: https://ai.google.dev/gemini-api/docs/video
    
    Supports:
    - Text-to-video
    - Image-to-video (first frame)
    - Seed for Veo 3 models (slightly improves determinism)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A cinematic shot of a majestic lion walking through the golden savannah at sunset."
                }),
                "model": ([
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                    "veo-3.0-generate-001",
                    "veo-3.0-fast-generate-001",
                    "veo-2.0-generate-001"
                ], {"default": "veo-3.1-generate-preview"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ([4, 5, 6, 7, 8], {"default": 8}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "person_generation": (["allow_all", "allow_adult", "dont_allow"], {"default": "allow_all"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),  # First frame / starting image
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "frame_count", "video_path", "video_uri")
    FUNCTION = "generate_video"
    CATEGORY = "AI API/Veo"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, prompt, model, aspect_ratio, duration_seconds, resolution, person_generation,
                   negative_prompt="", seed=0, image=None):
        """Return hash of inputs - same inputs = same hash = cached result"""
        import hashlib
        
        hash_parts = [
            str(prompt),
            str(model),
            str(aspect_ratio),
            str(duration_seconds),
            str(resolution),
            str(person_generation),
            str(negative_prompt or ""),
            str(seed),
        ]
        
        if image is not None:
            hash_parts.append(f"image:{image.shape}:{image.sum().item():.6f}")
        
        hash_string = "|".join(hash_parts)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def get_api_key(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                key = config.get("GEMINI_API_KEY", "")
                if key and key != "your_gemini_api_key_here":
                    return key
        return os.environ.get("GEMINI_API_KEY", "")

    def empty_output(self, msg=""):
        print(f"⚠️ Veo: Returning empty output - {msg}")
        empty_frames = torch.zeros((1, 720, 1280, 3), dtype=torch.float32)
        return (empty_frames, create_empty_audio(), 24.0, 0, msg, "")

    def generate_video(self, prompt, model, aspect_ratio, duration_seconds, resolution, person_generation,
                       negative_prompt="", seed=0, image=None):
        
        run_id = str(uuid.uuid4())[:8]
        
        api_key = self.get_api_key()
        if not api_key:
            return self.empty_output("API key missing - set GEMINI_API_KEY in config.json")
        
        print(f"\n{'='*60}")
        print(f"🎬 VEO VIDEO GENERATION [RUN: {run_id}]")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Aspect: {aspect_ratio}, Duration: {duration_seconds}s")
        print(f"Resolution: {resolution}, Person Generation: {person_generation}")
        print(f"Seed: {seed}")
        print(f"Image provided: {image is not None}")
        print(f"{'='*60}\n")
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Build config
            config_kwargs = {
                "aspect_ratio": aspect_ratio,
                "number_of_videos": 1,
                "person_generation": person_generation,
            }
            
            # Resolution - 1080p only supported for 8s duration with 16:9 aspect
            if resolution == "1080p":
                if duration_seconds == 8 and aspect_ratio == "16:9":
                    config_kwargs["resolution"] = resolution
                    print(f"📐 Using 1080p resolution")
                else:
                    print(f"⚠️ 1080p requires 8s duration and 16:9 aspect ratio, using 720p")
            
            if duration_seconds:
                config_kwargs["duration_seconds"] = int(duration_seconds)
            
            if negative_prompt and negative_prompt.strip():
                config_kwargs["negative_prompt"] = negative_prompt.strip()
            
            # Seed - available for Veo 3 models (slightly improves determinism)
            # Note: May not be supported in all SDK versions
            is_veo3 = "veo-3" in model or "veo-2" not in model
            if seed > 0 and is_veo3:
                try:
                    config_kwargs["seed"] = seed
                    print(f"🎲 Using seed: {seed}")
                except Exception as e:
                    print(f"⚠️ Seed not supported in this SDK version: {e}")
            
            print(f"📋 Config: {config_kwargs}")
            config = types.GenerateVideosConfig(**config_kwargs)
            
            # Generate video
            print(f"🚀 Starting video generation...")
            
            if image is not None:
                print(f"📷 Image-to-video mode (using image as first frame)")
                genai_image = tensor_to_genai_image(image)
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=genai_image,
                    config=config,
                )
            else:
                print(f"📝 Text-to-video mode")
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=config,
                )
            
            # Poll for completion
            poll_count = 0
            max_polls = 60
            
            while not operation.done:
                poll_count += 1
                elapsed = poll_count * 10
                print(f"⏳ Waiting for video generation... ({elapsed}s)")
                time.sleep(10)
                operation = client.operations.get(operation)
                
                if poll_count >= max_polls:
                    return self.empty_output("Timeout - generation took too long")
            
            print(f"✅ Generation complete after {poll_count * 10}s")
            
            if operation.error:
                error_msg = str(operation.error)
                print(f"❌ API Error: {error_msg}")
                return self.empty_output(f"API Error: {error_msg}")
            
            if not operation.response or not operation.response.generated_videos:
                print(f"❌ No videos in response")
                return self.empty_output("No videos generated")
            
            print(f"📦 Got {len(operation.response.generated_videos)} video(s)")
            
            generated_video = operation.response.generated_videos[0]
            
            # Capture the video URI for potential extension
            video_uri = getattr(generated_video.video, 'uri', '') or ''
            print(f"📎 Video URI: {video_uri}")
            
            print(f"📥 Downloading video...")
            client.files.download(file=generated_video.video)
            
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            video_filename = f"veo_{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            print(f"💾 Saving to: {video_path}")
            generated_video.video.save(video_path)
            
            if not os.path.exists(video_path):
                return self.empty_output("Video file was not saved")
            
            file_size = os.path.getsize(video_path)
            print(f"✅ Saved! File size: {file_size} bytes")
            
            if file_size < 1000:
                return self.empty_output(f"Video file too small ({file_size} bytes)")
            
            print(f"🎞️ Extracting frames and audio...")
            frames_tensor, audio, video_fps, frame_count = extract_frames_and_audio(video_path)
            
            if frames_tensor is None or frame_count == 0:
                return self.empty_output("Failed to extract frames from video")
            
            if audio is None:
                audio = create_empty_audio()
            
            print(f"\n{'='*60}")
            print(f"✅ VIDEO GENERATION COMPLETE")
            print(f"   Frames: {frame_count}")
            print(f"   FPS: {video_fps}")
            print(f"   Path: {video_path}")
            print(f"   URI: {video_uri}")
            print(f"{'='*60}\n")
            
            return (frames_tensor, audio, float(video_fps), frame_count, video_path, video_uri)
            
        except Exception as e:
            error_str = str(e)
            print(f"❌ Exception: {error_str}")
            import traceback
            traceback.print_exc()
            
            # If seed error, retry without seed
            if "seed" in error_str.lower() and seed > 0:
                print(f"⚠️ Retrying without seed...")
                return self.generate_video(prompt, model, aspect_ratio, duration_seconds,
                                          negative_prompt, 0, image)
            
            return self.empty_output(error_str)


class VeoVideoGeneratorAdvanced:
    """
    Advanced Veo 3.1 video generation with all features:
    - Reference images (up to 3)
    - First + Last frame interpolation
    - Full parameter control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A cinematic shot of a majestic lion walking through the golden savannah at sunset."
                }),
                "model": ([
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                ], {"default": "veo-3.1-generate-preview"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ([4, 6, 8], {"default": 8}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "person_generation": (["allow_all", "allow_adult", "dont_allow"], {"default": "allow_all"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "frame_count", "video_path", "video_uri")
    FUNCTION = "generate_video"
    CATEGORY = "AI API/Veo"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, prompt, model, aspect_ratio, duration_seconds, resolution, person_generation,
                   negative_prompt="", seed=0, first_frame=None, last_frame=None,
                   reference_image_1=None, reference_image_2=None, reference_image_3=None):
        import hashlib
        
        hash_parts = [
            str(prompt), str(model), str(aspect_ratio), str(duration_seconds),
            str(resolution), str(person_generation),
            str(negative_prompt or ""), str(seed),
        ]
        
        for name, img in [("first", first_frame), ("last", last_frame),
                          ("ref1", reference_image_1), ("ref2", reference_image_2),
                          ("ref3", reference_image_3)]:
            if img is not None:
                hash_parts.append(f"{name}:{img.shape}:{img.sum().item():.6f}")
        
        return hashlib.md5("|".join(hash_parts).encode()).hexdigest()
    
    def get_api_key(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                key = config.get("GEMINI_API_KEY", "")
                if key and key != "your_gemini_api_key_here":
                    return key
        return os.environ.get("GEMINI_API_KEY", "")

    def empty_output(self, msg=""):
        print(f"⚠️ Veo Advanced: Returning empty output - {msg}")
        empty_frames = torch.zeros((1, 720, 1280, 3), dtype=torch.float32)
        return (empty_frames, create_empty_audio(), 24.0, 0, msg, "")

    def generate_video(self, prompt, model, aspect_ratio, duration_seconds, resolution, person_generation,
                       negative_prompt="", seed=0, first_frame=None, last_frame=None,
                       reference_image_1=None, reference_image_2=None, reference_image_3=None):
        
        run_id = str(uuid.uuid4())[:8]
        
        api_key = self.get_api_key()
        if not api_key:
            return self.empty_output("API key missing")
        
        print(f"\n{'='*60}")
        print(f"🎬 VEO 3.1 ADVANCED VIDEO GENERATION [RUN: {run_id}]")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"First frame: {first_frame is not None}")
        print(f"Last frame: {last_frame is not None}")
        
        ref_count = sum(1 for r in [reference_image_1, reference_image_2, reference_image_3] if r is not None)
        print(f"Reference images: {ref_count}")
        print(f"Resolution: {resolution}, Person Generation: {person_generation}")
        print(f"Seed: {seed}")
        print(f"{'='*60}\n")
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Build config
            config_kwargs = {
                "aspect_ratio": aspect_ratio,
                "number_of_videos": 1,
                "duration_seconds": int(duration_seconds),
                "person_generation": person_generation,
            }
            
            # Resolution - 1080p only supported for 8s duration with 16:9 aspect
            if resolution == "1080p":
                if duration_seconds == 8 and aspect_ratio == "16:9":
                    config_kwargs["resolution"] = resolution
                    print(f"📐 Using 1080p resolution")
                else:
                    print(f"⚠️ 1080p requires 8s duration and 16:9 aspect ratio, using 720p")
            
            if negative_prompt and negative_prompt.strip():
                config_kwargs["negative_prompt"] = negative_prompt.strip()
            
            if seed > 0:
                config_kwargs["seed"] = seed
            
            # Last frame for interpolation
            if last_frame is not None:
                genai_last = tensor_to_genai_image(last_frame)
                config_kwargs["last_frame"] = genai_last
                print(f"📷 Using last frame for interpolation")
            
            # Reference images (up to 3) - Veo 3.1 only
            reference_images = []
            for i, ref_img in enumerate([reference_image_1, reference_image_2, reference_image_3], 1):
                if ref_img is not None:
                    genai_ref = tensor_to_genai_image(ref_img)
                    ref_obj = types.VideoGenerationReferenceImage(
                        image=genai_ref,
                        reference_type="asset"
                    )
                    reference_images.append(ref_obj)
                    print(f"📷 Reference image {i} added")
            
            if reference_images:
                config_kwargs["reference_images"] = reference_images
            
            print(f"📋 Config keys: {list(config_kwargs.keys())}")
            config = types.GenerateVideosConfig(**config_kwargs)
            
            # Generate video
            print(f"🚀 Starting video generation...")
            
            gen_kwargs = {
                "model": model,
                "prompt": prompt,
                "config": config,
            }
            
            # First frame
            if first_frame is not None:
                genai_first = tensor_to_genai_image(first_frame)
                gen_kwargs["image"] = genai_first
                print(f"📷 Using first frame as starting image")
            
            operation = client.models.generate_videos(**gen_kwargs)
            
            # Poll for completion
            poll_count = 0
            max_polls = 60
            
            while not operation.done:
                poll_count += 1
                print(f"⏳ Waiting... ({poll_count * 10}s)")
                time.sleep(10)
                operation = client.operations.get(operation)
                
                if poll_count >= max_polls:
                    return self.empty_output("Timeout")
            
            print(f"✅ Generation complete after {poll_count * 10}s")
            
            if operation.error:
                print(f"❌ Operation error: {operation.error}")
                return self.empty_output(f"API Error: {operation.error}")
            
            # Debug: print the full response
            print(f"📋 Operation response: {operation.response}")
            
            if not operation.response or not operation.response.generated_videos:
                # Check if there's a rai_media_filtered_reason (content filtered)
                if hasattr(operation.response, 'rai_media_filtered_reasons') and operation.response.rai_media_filtered_reasons:
                    print(f"⚠️ Content filtered: {operation.response.rai_media_filtered_reasons}")
                    return self.empty_output(f"Content filtered: {operation.response.rai_media_filtered_reasons}")
                print(f"❌ No videos in response. Full response: {operation.response}")
                return self.empty_output("No videos generated")
            
            generated_video = operation.response.generated_videos[0]
            
            # Capture the video URI for potential extension
            video_uri = getattr(generated_video.video, 'uri', '') or ''
            print(f"📎 Video URI: {video_uri}")
            
            print(f"📥 Downloading video...")
            client.files.download(file=generated_video.video)
            
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            video_path = os.path.join(output_dir, f"veo_adv_{timestamp}.mp4")
            
            generated_video.video.save(video_path)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
                return self.empty_output("Video save failed")
            
            print(f"💾 Saved: {video_path}")
            
            frames_tensor, audio, video_fps, frame_count = extract_frames_and_audio(video_path)
            
            if frames_tensor is None:
                return self.empty_output("Frame extraction failed")
            
            if audio is None:
                audio = create_empty_audio()
            
            print(f"✅ Complete: {frame_count} frames @ {video_fps} FPS")
            print(f"📎 Video URI for extension: {video_uri}")
            
            return (frames_tensor, audio, float(video_fps), frame_count, video_path, video_uri)
            
        except Exception as e:
            error_str = str(e)
            print(f"❌ Exception: {error_str}")
            import traceback
            traceback.print_exc()
            
            # If seed error, retry without seed
            if "seed" in error_str.lower() and seed > 0:
                print(f"⚠️ Retrying without seed...")
                return self.generate_video(prompt, model, aspect_ratio, duration_seconds,
                                          negative_prompt, 0, first_frame, last_frame,
                                          reference_image_1, reference_image_2, reference_image_3)
            
            return self.empty_output(error_str)


class VeoLoadVideo:
    """Load a video file and extract frames/audio."""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        files = [""]
        for d in [input_dir, output_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        files.append(f)
        
        return {
            "required": {
                "video_file": (sorted(set(files)), {"default": ""}),
            },
            "optional": {
                "video_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "frame_count", "video_path")
    FUNCTION = "load_video"
    CATEGORY = "AI API/Veo"
    
    @classmethod
    def IS_CHANGED(cls, video_file="", video_path=""):
        path = None
        if video_path and video_path.strip():
            path = video_path.strip()
        elif video_file:
            input_path = os.path.join(folder_paths.get_input_directory(), video_file)
            output_path = os.path.join(folder_paths.get_output_directory(), video_file)
            path = input_path if os.path.exists(input_path) else output_path
        
        if path and os.path.exists(path):
            return f"{path}_{os.path.getmtime(path)}_{os.path.getsize(path)}"
        return ""
    
    def load_video(self, video_file="", video_path=""):
        if video_path and video_path.strip():
            path = video_path.strip()
        elif video_file:
            input_path = os.path.join(folder_paths.get_input_directory(), video_file)
            output_path = os.path.join(folder_paths.get_output_directory(), video_file)
            path = input_path if os.path.exists(input_path) else output_path
        else:
            empty = torch.zeros((1, 720, 1280, 3))
            return (empty, create_empty_audio(), 24.0, 0, "")
        
        if not os.path.exists(path):
            print(f"❌ Video not found: {path}")
            empty = torch.zeros((1, 720, 1280, 3))
            return (empty, create_empty_audio(), 24.0, 0, "")
        
        print(f"📹 Loading video: {path}")
        
        frames_tensor, audio, video_fps, frame_count = extract_frames_and_audio(path)
        
        if frames_tensor is None:
            frames_tensor = torch.zeros((1, 720, 1280, 3))
            frame_count = 0
        
        if audio is None:
            audio = create_empty_audio()
        
        return (frames_tensor, audio, float(video_fps), frame_count, path)


class VeoVideoExtend:
    """
    Extend a previously generated Veo video.
    
    IMPORTANT: Connect the 'video_uri' output from VeoVideoGenerator directly to this node.
    The Veo API requires the ORIGINAL video URI from Gemini - re-uploading a downloaded 
    video file will NOT work for extension.
    
    Only works with Veo 3.1 models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_uri": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Continue the scene with smooth motion and consistent style."
                }),
                "model": ([
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                ], {"default": "veo-3.1-generate-preview"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "person_generation": (["allow_all", "allow_adult", "dont_allow"], {"default": "allow_all"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "frame_count", "video_path", "video_uri")
    FUNCTION = "extend_video"
    CATEGORY = "AI API/Veo"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, video_uri, prompt, model, resolution, person_generation,
                   negative_prompt=""):
        import hashlib
        
        hash_parts = [
            str(video_uri), str(prompt), str(model),
            str(resolution), str(person_generation),
            str(negative_prompt or ""),
        ]
        
        return hashlib.md5("|".join(hash_parts).encode()).hexdigest()
    
    def get_api_key(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                key = config.get("GEMINI_API_KEY", "")
                if key and key != "your_gemini_api_key_here":
                    return key
        return os.environ.get("GEMINI_API_KEY", "")

    def empty_output(self, msg=""):
        print(f"⚠️ Veo Extend: Returning empty output - {msg}")
        empty_frames = torch.zeros((1, 720, 1280, 3), dtype=torch.float32)
        return (empty_frames, create_empty_audio(), 24.0, 0, msg, "")

    def extend_video(self, video_uri, prompt, model, resolution, person_generation,
                     negative_prompt=""):
        
        run_id = str(uuid.uuid4())[:8]
        
        api_key = self.get_api_key()
        if not api_key:
            return self.empty_output("API key missing")
        
        if not video_uri or not video_uri.strip():
            return self.empty_output("No video_uri provided. Connect the 'video_uri' output from VeoVideoGenerator.")
        
        video_uri = video_uri.strip()
        
        # Validate it looks like a Gemini file URI
        if not video_uri.startswith("https://generativelanguage.googleapis.com/"):
            return self.empty_output(f"Invalid video_uri. Must be a Gemini file URI starting with 'https://generativelanguage.googleapis.com/'. Got: {video_uri[:50]}...")
        
        print(f"\n{'='*60}")
        print(f"🎬 VEO VIDEO EXTENSION [RUN: {run_id}]")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Source video URI: {video_uri}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Resolution: {resolution}, Person Generation: {person_generation}")
        print(f"{'='*60}\n")
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Create video object using the original video URI directly
            # No upload needed - we use the original Gemini file URI
            video_obj = types.Video(uri=video_uri)
            print(f"📎 Using video URI directly: {video_uri}")
            
            # Build config
            config_kwargs = {
                "number_of_videos": 1,
                "person_generation": person_generation,
            }
            
            # Resolution
            if resolution == "1080p":
                config_kwargs["resolution"] = resolution
                print(f"📐 Using 1080p resolution")
            
            if negative_prompt and negative_prompt.strip():
                config_kwargs["negative_prompt"] = negative_prompt.strip()
            
            # Note: seed is NOT supported for video extension in Gemini API
            
            print(f"📋 Config keys: {list(config_kwargs.keys())}")
            config = types.GenerateVideosConfig(**config_kwargs)
            
            # Generate extended video
            print(f"🚀 Starting video extension...")
            
            operation = client.models.generate_videos(
                model=model,
                prompt=prompt,
                video=video_obj,
                config=config,
            )
            
            # Poll for completion
            poll_count = 0
            max_polls = 60
            
            while not operation.done:
                poll_count += 1
                print(f"⏳ Waiting... ({poll_count * 10}s)")
                time.sleep(10)
                operation = client.operations.get(operation)
                
                if poll_count >= max_polls:
                    return self.empty_output("Timeout")
            
            print(f"✅ Extension complete after {poll_count * 10}s")
            
            if operation.error:
                print(f"❌ Operation error: {operation.error}")
                return self.empty_output(f"API Error: {operation.error}")
            
            # Debug: print the full response
            print(f"📋 Operation response: {operation.response}")
            
            if not operation.response or not operation.response.generated_videos:
                if hasattr(operation.response, 'rai_media_filtered_reasons') and operation.response.rai_media_filtered_reasons:
                    print(f"⚠️ Content filtered: {operation.response.rai_media_filtered_reasons}")
                    return self.empty_output(f"Content filtered: {operation.response.rai_media_filtered_reasons}")
                print(f"❌ No videos in response. Full response: {operation.response}")
                return self.empty_output("No videos generated")
            
            generated_video = operation.response.generated_videos[0]
            
            # Capture the new video URI for potential further extension
            new_video_uri = getattr(generated_video.video, 'uri', '') or ''
            print(f"📎 New video URI: {new_video_uri}")
            
            print(f"📥 Downloading extended video...")
            client.files.download(file=generated_video.video)
            
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            extended_path = os.path.join(output_dir, f"veo_extended_{timestamp}.mp4")
            
            generated_video.video.save(extended_path)
            
            if not os.path.exists(extended_path) or os.path.getsize(extended_path) < 1000:
                return self.empty_output("Video save failed")
            
            print(f"💾 Saved: {extended_path}")
            
            frames_tensor, audio, video_fps, frame_count = extract_frames_and_audio(extended_path)
            
            if frames_tensor is None:
                return self.empty_output("Failed to extract frames")
            
            if audio is None:
                audio = create_empty_audio()
            
            print(f"🎉 Extended video: {frame_count} frames @ {video_fps:.2f}fps")
            print(f"📎 Video URI for further extension: {new_video_uri}")
            return (frames_tensor, audio, float(video_fps), frame_count, extended_path, new_video_uri)
            
        except Exception as e:
            import traceback
            print(f"❌ Error extending video: {e}")
            traceback.print_exc()
            return self.empty_output(f"Error: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "VeoVideoGenerator": VeoVideoGenerator,
    "VeoVideoGeneratorAdvanced": VeoVideoGeneratorAdvanced,
    "VeoLoadVideo": VeoLoadVideo,
    "VeoVideoExtend": VeoVideoExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoVideoGenerator": "Veo Video Generator",
    "VeoVideoGeneratorAdvanced": "Veo Video Generator (Advanced)",
    "VeoLoadVideo": "Veo Load Video",
    "VeoVideoExtend": "Veo Video Extend",
}
