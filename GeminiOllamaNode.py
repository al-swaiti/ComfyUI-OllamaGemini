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
        "Veo3-TextToVideo": (
            "Create an optimized Veo 3.1 text-to-video prompt following Google's official prompting guide. "
            "Structure your prompt with these essential elements in order: "
            "(1) COMPOSITION & CAMERA: Start with shot type (close-up, wide shot, medium shot, extreme close-up) and camera motion (dolly, tracking, pan, aerial view, POV shot, drone view). "
            "(2) SUBJECT: Clearly define the main subject with specific details - person, animal, object, or scene with descriptive attributes. "
            "(3) ACTION: Describe what the subject is doing with dynamic verbs and temporal flow. "
            "(4) CONTEXT & ENVIRONMENT: Establish the setting, location, time of day, and atmospheric conditions. "
            "(5) STYLE & AMBIANCE: Specify visual style (cinematic, film noir, documentary, animated), color tones (warm, cool, muted), and lighting (golden hour, dramatic side lighting, neon glow). "
            "(6) AUDIO CUES (Veo 3 native audio): Include dialogue in quotes ('She whispers, \"Follow me.\"'), sound effects (footsteps on gravel, wind howling), and ambient sounds (distant traffic, birds chirping). "
            "IMPORTANT TIPS: Use descriptive adjectives and adverbs. Avoid instructive language like 'no' or 'don't' - describe what you WANT. "
            "Keep prompt between 50-150 words for optimal results. Be specific about facial details for portraits. "
            "Return ONLY the video prompt as a flowing, cinematic paragraph."
        ),

        "Veo3-ReferenceImages": (
            "Create an optimized Veo 3.1 reference image video prompt. Reference images preserve subject appearance (person, character, product) in the generated video. "
            "CRITICAL: Your prompt must explicitly describe the reference image subjects and how they appear in the video. "
            "Structure your prompt: "
            "(1) SUBJECT IDENTIFICATION: Clearly describe each referenced subject from your images - be specific about their appearance, clothing, accessories, and distinguishing features that match your reference images. "
            "(2) SCENE SETUP: Establish the environment where your referenced subjects will appear - location, setting, atmosphere. "
            "(3) ACTION & MOVEMENT: Describe what the referenced subjects are doing - their movements, interactions, expressions. Use cinematic language. "
            "(4) CAMERA & COMPOSITION: Specify shot type (medium shot, wide shot, close-up), camera movement (slow pan, tracking shot, dolly), and framing. "
            "(5) STYLE & MOOD: Define the visual aesthetic, color palette, lighting conditions, and emotional tone. "
            "(6) INTEGRATION: Describe how multiple referenced elements interact and relate to each other spatially. "
            "EXAMPLE STRUCTURE: 'The video opens with a [shot type] of [subject from reference 1 with detailed description]. [Subject] [action] while wearing [clothing/accessories from reference 2]. The scene [camera movement] to reveal [environment]. [Lighting/mood description].' "
            "NOTE: Reference images work best for single subjects (person, character, product) that you want to preserve consistently. Maximum 3 reference images. Duration must be 8 seconds with 16:9 aspect ratio. "
            "Return ONLY the video prompt describing how your referenced subjects appear and act."
        ),

        "Veo3-Interpolation": (
            "Create an optimized Veo 3.1 first-to-last frame interpolation prompt. This feature generates video that transitions between your specified start and end images. "
            "CRITICAL: Your prompt must describe the TRANSFORMATION and JOURNEY between the two frames - what happens, how it changes, the motion path. "
            "Structure your prompt: "
            "(1) STARTING STATE: Briefly acknowledge the initial scene/subject position from your first frame image. "
            "(2) TRANSFORMATION DESCRIPTION: This is the KEY element - describe the journey, motion, or change that occurs between frames. Use temporal language: 'slowly transitions,' 'gradually transforms,' 'smoothly morphs,' 'progressively moves.' "
            "(3) MOTION PATH: Describe HOW the subject moves or changes - direction, speed, style of movement (graceful, dramatic, subtle). "
            "(4) ATMOSPHERIC EVOLUTION: Describe any changes in lighting, mood, or environment during the transition. "
            "(5) ENDING STATE: Reference the destination/final state shown in your last frame image. "
            "(6) CINEMATIC STYLE: Include camera behavior, visual style, and emotional tone of the transition. "
            "EXAMPLE PATTERNS: "
            "- Morphing: 'A cinematic transformation as [subject] slowly morphs from [state A] into [state B], the change rippling across...' "
            "- Movement: 'Smooth tracking shot following [subject] as they journey from [location A] to [location B], passing through...' "
            "- Time-lapse: 'The scene gradually shifts from [time/state A] to [time/state B], with [elements] slowly changing...' "
            "NOTE: Interpolation requires both first frame (image parameter) and last frame (last_frame config). Duration is fixed at 8 seconds. Works with both 16:9 and 9:16 aspect ratios. "
            "Return ONLY the interpolation prompt describing the transformation journey between your two frames."
        ),

        "VideoGen": "Create a professional cinematic video generation prompt based on my description. Structure your prompt in this precise order: (1) SUBJECT: Define main character(s)/object(s) with specific, vivid details (appearance, expressions, attributes); (2) CONTEXT/SCENE: Establish the detailed environment with atmosphere, time of day, weather, and spatial relationships; (3) ACTION: Describe precise movements and temporal flow using dynamic verbs and sequential language ('first... then...'); (4) CINEMATOGRAPHY: Specify exact camera movements (dolly, pan, tracking), shot types (close-up, medium, wide), lens choice (35mm, telephoto), and professional lighting terminology (Rembrandt, golden hour, backlit); (5) STYLE: Define the visual aesthetic using specific references to film genres, directors, or animation styles. For realistic scenes, emphasize photorealism with natural lighting and physics. For abstract/VFX, include stylistic terms (surreal, psychedelic) and dynamic descriptors (swirling, morphing). For animation, specify the exact style (anime, 3D cartoon, hand-drawn). Craft a single cohesive paragraph that flows naturally while maintaining technical precision. Return ONLY the prompt text itself no more 200 tokens.",

        "FLUX.1-dev": "As an elite text-to-image prompt engineer, craft an exceptional FLUX.1-dev prompt from my description. Create a hyper-detailed, cinematographic paragraph that includes: (1) precise subject characterization with emotional undertones, (2) specific artistic influences from legendary painters/photographers, (3) technical camera specifications (lens, aperture, perspective), (4) sophisticated lighting setup with exact quality and direction, (5) atmospheric elements and depth effects, (6) composition techniques, and (7) post-processing styles. Use language that balances technical precision with artistic vision. Return ONLY the prompt text itself - no explanations or formatting no more 200 tokens.",

        "FLUX.2-dev": (
            "As a FLUX.2-dev prompt specialist, craft an optimized prompt following the official BFL prompting guide. "
            "IMPORTANT: FLUX.2 does NOT support negative prompts - describe only what you want, not what to avoid. "
            "Structure using: Subject + Action + Style + Context (priority order - most important elements first). "
            "(1) SUBJECT: Main focus with precise details - who/what is in the image with specific attributes. "
            "(2) ACTION/POSE: What the subject is doing or their positioning. "
            "(3) STYLE: Artistic approach using specific references - for photorealism specify camera model, lens (e.g., 'shot on Sony A7IV, 85mm f/1.8'), film stock (e.g., 'Kodak Portra 400'), or era style ('80s vintage photo', '2000s digicam'). "
            "(4) CONTEXT: Setting, lighting conditions (golden hour, studio lighting, Rembrandt lighting), time of day, mood, and atmospheric conditions. "
            "(5) TECHNICAL: Camera angle, depth of field, composition (rule of thirds), and quality descriptors. "
            "(6) TEXT (if needed): Use quotation marks for text, specify placement ('top-right corner'), style ('bold serif', 'neon letters'), and color (use hex codes like '#FF5733' for precision). "
            "Keep prompt 30-80 words for optimal results. Put most important elements first. "
            "For hex colors, associate them with specific objects: 'The vase has color #02eb3c'. "
            "Return ONLY the prompt text itself - no explanations, no negative prompts, under 200 tokens."
        ),

        "FLUX.2-dev-Edit": (
            "As a FLUX.2-dev image editing specialist, craft precise editing instructions following the official BFL guide for multi-reference editing. "
            "FLUX.2 combines generation and editing in one model with up to 10 reference images. "
            "Structure your editing prompt: "
            "(1) ACTION: Clear verb describing the change (replace, add, remove, modify, transform, combine, style-transfer). "
            "(2) TARGET: Specific element to edit using descriptive language (the blue ceramic vase, the person in red jacket). "
            "(3) RESULT: Detailed description of desired outcome with visual specifics. "
            "(4) PRESERVATION: What should remain unchanged (keep the background lighting, maintain facial features, preserve the original composition). "
            "(5) STYLE MATCHING: Ensure edits blend with original - match lighting, perspective, color palette, and artistic style. "
            "(6) INTEGRATION: Describe how new elements should relate spatially and stylistically to existing content. "
            "For character/product consistency across images, maintain detailed descriptions in every prompt. "
            "For text changes: use quotes for new text, specify typography preservation. "
            "For color precision: use hex codes associated with specific elements. "
            "Keep instructions conversational and descriptive - FLUX.2 understands context. "
            "Return ONLY the editing instruction, under 100 tokens."
        ),

        "FLUX.2-dev-JSON": (
            "Create a structured JSON prompt optimized for FLUX.2-dev complex scenes. "
            "Use this schema for precise control: "
            '{"scene": "overall description", '
            '"subjects": [{"description": "detailed subject", "position": "where in frame", "action": "what doing", "color_palette": ["colors"]}], '
            '"style": "artistic style with camera/film references", '
            '"color_palette": ["#hex1", "#hex2"], '
            '"lighting": "specific lighting setup", '
            '"mood": "emotional tone", '
            '"background": "background details", '
            '"composition": "framing technique", '
            '"camera": {"angle": "angle", "lens": "lens mm", "f-number": "aperture", "focus": "focus behavior"}}. '
            "Associate hex colors with specific objects. Include camera model and lens for photorealism. "
            "For typography: add text field with content in quotes, placement, and style. "
            "Return ONLY valid JSON - no explanations."
        ),

        "SDXL": "Create a premium comma-separated tag prompt for SDXL based on my description. Structure the prompt with these elements in order of importance: (1) main subject with precise descriptors, (2) high-impact artistic medium (oil painting, digital art, photography, etc.), (3) specific art movement or style with named influences, (4) professional lighting terminology (rembrandt, cinematic, golden hour, etc.), (5) detailed environment/setting, (6) exact camera specifications (35mm, telephoto, macro, etc.), (7) composition techniques, (8) color palette/mood, and (9) post-processing effects. Use 20-30 tags maximum, prioritizing quality descriptors over quantity. Include 2-3 relevant artist references whose style matches the desired aesthetic. Return ONLY the comma-separated tags without explanations or formatting.",

        "FLUXKontext": "Generate precise Flux Kontext editing instructions using this complete framework: (1) clear action verbs (change, add, remove, replace, transform, modify) with specific target objects and spatial identifiers, (2) detailed modification specs with quantified values (percentages, measurements, exact colors), (3) character consistency protection using 'while maintaining exact same character/facial features/identity' - critical for character preservation, (4) multi-layered preservation clauses for composition/lighting/atmosphere/positioning, (5) specific descriptors avoiding all pronouns - use detailed physical attributes, (6) precise style names with medium characteristics and cultural context, (7) quoted text replacements with typography preservation, (8) semantic relationship maintenance for natural blending, (9) context-aware modifications that understand whole image content. Focus on descriptive language over complex formatting. Ensure edits blend seamlessly with existing content through contextual understanding. Return ONLY the Flux Kontext instruction no more 50 words.",

        "Imagen4": (
            "Craft a vivid, layered image prompt optimized for Imagen 4. "
            "Structure in this precise order: "
            "(1) SUBJECT: Define the main subject(s) with concrete, vivid traits (appearance, pose, expression). "
            "(2) SCENE: Establish environment and context (setting, time of day, background elements). "
            "(3) ATMOSPHERE & LIGHT: Specify lighting—natural or artificial (e.g. golden hour, dramatic side lighting), and mood. "
            "(4) COMPOSITION & TECHNICAL: Describe camera angle, framing, lens effect (e.g. shallow depth of field, wide-angle), perspective and spatial arrangement. "
            "(5) STYLE & QUALITY: Include artistic style and medium (photorealistic, oil painting, illustration), text layout if needed, resolution cues (e.g. 2K, high resolution), and mood-enhancing terms (cinematic, hyper-realistic). "
            "Aim for specificity and clarity; layer in descriptive detail progressively. "
            "Return ONLY the prompt text itself, in a cohesive single paragraph, under 200 tokens."
        ),

        "GeminiNanaBananaEdit": (
            "Convert my editing request into precise Gemini conversational image editing instructions that leverage its mask-free contextual editing capabilities: "
            "(1) CONTEXTUAL REFERENCE: Begin with 'Using the provided image' and identify the specific element to modify using detailed descriptive language rather than spatial coordinates (the blue ceramic vase on the wooden table, the person wearing the red jacket in the center). "
            "(2) EDIT ACTION: Use clear, conversational verbs that specify the transformation (replace with, transform into, add beside, remove while preserving, change the color to, adjust the lighting to make more). "
            "(3) INTEGRATION SPECIFICATION: Describe how the change should blend seamlessly with existing elements, maintaining consistency in lighting, perspective, style, and atmosphere (ensuring the new element matches the existing warm golden hour lighting and rustic kitchen aesthetic). "
            "(4) PRESERVATION DIRECTIVES: Explicitly state what should remain unchanged to protect critical elements (keep everything else exactly the same, preserve the original character's facial features and expression, maintain the architectural details of the background). "
            "(5) STYLE CONTINUITY: Reference the existing visual style and ensure the modification matches (in the same photorealistic style, maintaining the impressionistic brushwork quality, keeping the vintage film photography aesthetic). "
            "(6) RELATIONSHIP CONTEXT: Describe how the edited element should relate to other objects in the scene for natural composition (positioned naturally beside the existing furniture, scaled appropriately for a person of that height, casting realistic shadows on the ground). "
            "Structure as conversational instructions under 75 words that feel like natural directions to an artist who can see and understand the full image context. "
            "Avoid technical jargon and focus on descriptive, intuitive language that leverages Gemini's contextual understanding. "
            "Return ONLY the editing instruction text."
        ),

        "NanaBananaPro": (
            "As a Nano Banana Pro (Gemini 3 Pro Image) prompt specialist, craft an optimized text-to-image prompt following official Google prompting guidelines. "
            "CORE PRINCIPLE: Describe the scene narratively as a cohesive paragraph - NOT as keyword lists. Gemini's strength is deep language understanding. "
            "Structure your prompt with these elements: "
            "(1) SUBJECT & INTENT: Define the main subject with hyper-specific details (instead of 'fantasy armor' say 'ornate elven plate armor, etched with silver leaf patterns, with a high collar and pauldrons shaped like falcon wings'). Explain the purpose/context of the image. "
            "(2) SCENE & ENVIRONMENT: Establish the setting, time of day, atmosphere, and spatial relationships. Use step-by-step layering for complex scenes ('First, create a misty forest at dawn. Then, add a moss-covered stone altar in the foreground. Finally, place a glowing sword on the altar'). "
            "(3) LIGHTING & MOOD: Specify lighting type (studio softbox, golden hour, Rembrandt lighting, dramatic side lighting), quality, and emotional atmosphere. "
            "(4) CAMERA & COMPOSITION: Use photographic terms - shot type (wide-angle, macro, close-up portrait), lens details (85mm f/1.4), camera angle (low-angle perspective, bird's eye view), focus behavior (shallow depth of field, bokeh effect). "
            "(5) STYLE & QUALITY: Define artistic style (photorealistic, oil painting, anime, minimalist), reference specific aesthetics, include resolution cues (high resolution, studio quality). "
            "(6) TEXT RENDERING (if needed): For text in images, specify exact text in quotes, font style descriptively ('elegant serif calligraphy', 'bold sans-serif'), placement, and language. "
            "Use 'semantic negative prompts' - describe what you WANT, not what to avoid (say 'an empty, deserted street' instead of 'no cars'). "
            "Return ONLY the prompt as a flowing, descriptive paragraph under 200 tokens."
        ),

        "NanaBananaPro-Edit": (
            "As a Nano Banana Pro image editing specialist, craft precise conversational editing instructions optimized for Gemini 3 Pro Image's advanced mask-free editing capabilities. "
            "Gemini excels at understanding context and can edit images through natural language without masks or coordinates. "
            "Structure your editing instruction: "
            "(1) ELEMENT IDENTIFICATION: Describe the specific element to modify using rich visual descriptors, not coordinates (the vintage wooden bookshelf on the left wall, the woman in the blue floral dress standing near the window). "
            "(2) EDIT ACTION: Use clear, conversational verbs - add, remove, replace, transform, change, adjust, modify, extend, reduce. Be specific about the transformation. "
            "(3) DESIRED OUTCOME: Describe the end result in detail with visual specifics (transform the day scene into a cozy evening with warm lamplight casting soft shadows). "
            "(4) STYLE MATCHING: Ensure edits blend seamlessly - match existing lighting conditions, color palette, artistic style, perspective, and atmosphere. Reference the original's visual qualities. "
            "(5) PRESERVATION CLAUSE: Explicitly state what must remain unchanged (keep the character's facial features exactly the same, preserve the background architecture, maintain the original composition). "
            "(6) REFINEMENT NOTES: For iterative editing, use natural follow-ups ('That's great, but make the lighting warmer' or 'Keep everything the same but change the expression to more serious'). "
            "For LOCALIZED EDITING: Select, refine, and transform specific parts - adjust camera angles, change focus, apply color grading, transform lighting (day to night, add bokeh). "
            "For MULTI-IMAGE COMPOSITION: When combining elements from multiple images, describe how elements should integrate spatially and stylistically. "
            "Write as natural, conversational instructions an artist would understand. Under 100 tokens. "
            "Return ONLY the editing instruction."
        ),

        "NanaBananaPro-Pro": (
            "As a Nano Banana Pro professional asset production specialist, create a studio-quality prompt leveraging Gemini 3 Pro Image's advanced capabilities. "
            "This template is optimized for: professional mockups, marketing assets, infographics, storyboards, and high-fidelity productions up to 4K. "
            "Structure with professional precision: "
            "(1) ASSET TYPE & PURPOSE: Define the deliverable type (product mockup, infographic, storyboard panel, logo design, marketing banner) and its intended use. "
            "(2) SUBJECT SPECIFICATION: Hyper-detailed subject description with exact attributes. For products: materials, colors, branding elements. For characters: detailed physical features for consistency across multiple generations. "
            "(3) COMPOSITION & LAYOUT: Professional framing - rule of thirds, negative space for text overlay, visual hierarchy. Specify aspect ratio (1:1, 16:9, 9:16, 4:3, 21:9) and resolution (1K, 2K, 4K). "
            "(4) PROFESSIONAL LIGHTING: Studio-grade lighting setups (three-point softbox, rim lighting, product photography lighting with gradient backgrounds). "
            "(5) TEXT & TYPOGRAPHY (critical for Nano Banana Pro): Specify exact text in quotes, font style description ('modern minimalist sans-serif', 'elegant script calligraphy'), size hierarchy, placement, and color. For multilingual: specify language. "
            "(6) BRAND CONSISTENCY: Reference brand colors, style guides, maintain visual identity across assets. For character consistency: include detailed descriptions in every prompt. "
            "(7) GROUNDING (optional): For real-time data visualization (weather, sports scores, stock charts), mention the data source context. "
            "For STORYBOARDS: Specify panel layout, shot types (establishing, medium, close-up, POV), and sequential flow. "
            "For INFOGRAPHICS: Structure information hierarchy, use clear data visualization principles. "
            "Return ONLY the professional prompt, structured as a clear brief, under 250 tokens."
        )
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
                    "Veo3-TextToVideo",
                    "Veo3-ReferenceImages",
                    "Veo3-Interpolation",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "FLUX.2-dev",
                    "FLUX.2-dev-Edit",
                    "FLUX.2-dev-JSON",
                    "SDXL",
                    "FLUXKontext",
                    "Imagen4",
                    "GeminiNanaBananaEdit",
                    "NanaBananaPro",
                    "NanaBananaPro-Edit",
                    "NanaBananaPro-Pro"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Qwen"

    def generate_content(self, prompt, qwen_model, max_tokens, temperature, top_p, structure_output, prompt_structure, structure_format, output_format, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None):
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

            # Handle multiple images
            all_images = [image1, image2, image3, image4, image5]
            provided_images = [img for img in all_images if img is not None]

            if provided_images:
                content = [{"type": "text", "text": modified_prompt}]
                for img in provided_images:
                    image_b64 = self.tensor_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
                messages.append({"role": "user", "content": content})
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

        # Handle multiple images
        all_images = [image1, image2, image3, image4, image5]
        provided_images = [img for img in all_images if img is not None]

        if provided_images:
            content = [{"type": "text", "text": modified_prompt}]
            for img in provided_images:
                image_b64 = tensor_to_base64(img)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": modified_prompt}]

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

            return (textoutput,)

        except Exception as e:
            return (f"API Error: {str(e)}",)

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
                    "Veo3-TextToVideo",
                    "Veo3-ReferenceImages",
                    "Veo3-Interpolation",
                    "HunyuanVideo",
                    "Wan2.1",
                    "FLUX.1-dev",
                    "FLUX.2-dev",
                    "FLUX.2-dev-Edit",
                    "FLUX.2-dev-JSON",
                    "SDXL",
                    "FLUXKontext",
                    "Imagen4",
                    "GeminiNanaBananaEdit",
                    "NanaBananaPro",
                    "NanaBananaPro-Edit",
                    "NanaBananaPro-Pro"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Claude"

    def generate_content(self, prompt, model, max_tokens, structure_output, prompt_structure, structure_format, output_format, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None):
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

        # Handle multiple images
        all_images = [image1, image2, image3, image4, image5]
        provided_images = [img for img in all_images if img is not None]

        if provided_images:
            content = [{"type": "text", "text": modified_prompt}]
            for img in provided_images:
                image_b64 = tensor_to_base64(img)
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": modified_prompt}]

        try:
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
                    "Veo3-TextToVideo",
                    "Veo3-ReferenceImages",
                    "Veo3-Interpolation",
                    "VideoGen",
                    "FLUX.1-dev",
                    "FLUX.2-dev",
                    "FLUX.2-dev-Edit",
                    "FLUX.2-dev-JSON",
                    "SDXL",
                    "FLUXKontext",
                    "Imagen4",
                    "GeminiNanaBananaEdit",
                    "NanaBananaPro",
                    "NanaBananaPro-Edit",
                    "NanaBananaPro-Pro"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video": ("IMAGE",),  # Video is represented as a tensor with shape [frames, height, width, channels]
                "audio": ("AUDIO",),  # Audio input
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Gemini"

    def generate_content(self, prompt, input_type, gemini_model, stream, structure_output, prompt_structure, structure_format, output_format, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None, video=None, audio=None):
        if api_key:
            update_config_key("GEMINI_API_KEY", api_key)
            self.gemini_api_key = api_key
            genai.configure(api_key=self.gemini_api_key, transport='rest')

        if not self.gemini_api_key:
            return ("Gemini API key missing. Please provide it in the node's api_key input.",)

        try:
            generation_config = {"temperature": 0.7, "top_p": 0.8, "top_k": 40}
            modified_prompt = apply_prompt_template(prompt, prompt_structure)
            if structure_output:
                print(f"Requesting structured output from {gemini_model}")
                modified_prompt = f"{modified_prompt}\n\n{structure_format}"
                print(f"Modified prompt with structure format")

            model = genai.GenerativeModel(gemini_model)

            content = [modified_prompt]

            if input_type == "image":
                # Handle multiple images
                all_images = [image1, image2, image3, image4, image5]
                provided_images = [img for img in all_images if img is not None]

                if provided_images:
                    for img in provided_images:
                        pil_image = tensor_to_pil_image(img)
                        content.append(pil_image)
            elif input_type == "video" and video is not None:
                frames = sample_video_frames(video)
                if frames:
                    content.extend(frames)
                else:
                    return ("Error: Could not extract frames from video",)
            # Audio processing would go here if the modern library supported it this way

            print(f"Sending request to Gemini API with model: {gemini_model}")

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = model.generate_content(
                content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=stream
            )

            if stream:
                textoutput = "".join([chunk.text for chunk in response if hasattr(chunk, 'text')])
            else:
                if not hasattr(response, 'text'):
                    if hasattr(response, 'prompt_feedback'):
                        return (f"API Error: Content blocked - {response.prompt_feedback}",)
                    else:
                        return (f"API Error: Empty response from Gemini API",)
                textoutput = response.text

            print("Gemini API response received successfully")

            if textoutput.strip():
                clean_text = textoutput.strip()
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break
                if output_format == "json":
                    try:
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"
                        json_output = json.dumps({key_name: clean_text}, indent=2)
                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
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
                    "Veo3-TextToVideo",
                    "Veo3-ReferenceImages",
                    "Veo3-Interpolation",
                    "VideoGen",
                    "FLUX.1-dev",
                    "FLUX.2-dev",
                    "FLUX.2-dev-Edit",
                    "FLUX.2-dev-JSON",
                    "SDXL",
                    "FLUXKontext",
                    "Imagen4",
                    "GeminiNanaBananaEdit",
                    "NanaBananaPro",
                    "NanaBananaPro-Edit",
                    "NanaBananaPro-Pro"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video": ("IMAGE",),  # Video is represented as a tensor with shape [frames, height, width, channels]
                "audio": ("AUDIO",),  # Audio input
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Ollama"

    def generate_content(self, prompt, input_type, ollama_model, keep_alive, structure_output, prompt_structure, structure_format, output_format, image1=None, image2=None, image3=None, image4=None, image5=None, video=None, audio=None, seed=0):
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
            "keep_alive": f"{keep_alive}m",
            "options": {"seed": seed}
        }

        try:
            # Process different input types
            if input_type == "text":
                # Text-only input, no additional processing needed
                print(f"Processing text input for Ollama API")

            elif input_type == "image":
                # Handle multiple images
                all_images = [image1, image2, image3, image4, image5]
                provided_images = [img for img in all_images if img is not None]

                if provided_images:
                    print(f"Processing {len(provided_images)} image(s) for Ollama API")
                    image_data = []
                    for img in provided_images:
                        pil_image = tensor_to_pil_image(img)
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        image_data.append(base64.b64encode(buffered.getvalue()).decode())

                    payload["images"] = image_data
                    # Update prompt to indicate image analysis
                    modified_prompt = f"Analyze these image(s): {modified_prompt}"
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
    # "OpenAIAPI": GeminiOpenAIAPI,  # Disabled - class is broken
    "ClaudeAPI": GeminiClaudeAPI,
    "QwenAPI": GeminiQwenAPI,
    "GeminiTextSplitter": GeminiTextSplitByDelimiter,
    "GeminiSaveText": GeminiSaveTextFile,
    "ListAvailableModels": GeminiListAvailableModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAPI": "Gemini API",
    "OllamaAPI": "Ollama API",
    # "OpenAIAPI": "OpenAI API",  # Disabled - class is broken
    "ClaudeAPI": "Claude API",
    "QwenAPI": "Qwen API",
    "GeminiTextSplitter": "Gemini Text Splitter",
    "GeminiSaveText": "Gemini Save Text",
    "ListAvailableModels": "List Available Models",
}
