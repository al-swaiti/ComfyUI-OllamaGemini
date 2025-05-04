import os
import json
import random
import time
import pathlib
import uuid
import base64
from collections import defaultdict
import importlib.util
from io import BytesIO
from PIL import Image
import numpy as np

# Try to import required libraries, but don't fail if not available
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Torch or torchaudio not available. Video and audio processing will be disabled.")

# Try to import AI model libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI module not available.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI module not available.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic (Claude) module not available.")

try:
    import dashscope
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Qwen module (dashscope) not available.")

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Requests module not available, Ollama integration disabled.")

from .prompt_stylerx import StylerData, Template

class GeminiSmartPromptGenerator:
    """
    A node that intelligently combines multiple style elements from different categories
    and uses various AI models to create enhanced, creative prompts.
    """
    def __init__(self):
        # Load all style data
        self.styler_data = StylerData()
        
        # Get API keys and configuration
        self.config = self.load_config()
        
        # Configure AI models
        self.setup_ai_models()
        
        # Load style categories for the selectors
        self.style_categories = self.get_style_categories()

        # Initialize a run counter to ensure different results on each run
        self.run_counter = 0

        # Random base prompt subjects
        self.random_subjects = [
            "landscape", "portrait", "cityscape", "seascape", "still life",
            "character", "fantasy creature", "robot", "vehicle", "animal",
            "monster", "superhero", "villain", "warrior", "wizard",
            "spaceship", "building", "castle", "temple", "forest", "mountain",
            "beach", "desert", "jungle", "underwater scene", "space scene",
            "futuristic city", "medieval town", "cyberpunk street", "steampunk workshop",
            "fantasy world", "alien planet", "dragon", "phoenix", "unicorn", "mermaid",
            "ghost", "vampire", "werewolf", "fairy", "elf", "dwarf", "orc", "goblin",
            "angel", "demon", "god", "goddess", "titan", "giant", "colossus"
        ]

        # Random adjectives to enhance prompts
        self.random_adjectives = [
            "ancient", "futuristic", "mysterious", "magical", "dark", "bright",
            "glowing", "ethereal", "majestic", "terrifying", "beautiful", "ugly",
            "corrupted", "pure", "mechanical", "organic", "crystalline", "fiery",
            "frozen", "cosmic", "divine", "demonic", "elegant", "primitive",
            "colorful", "monochromatic", "vibrant", "muted", "translucent", "iridescent",
            "weathered", "pristine", "ornate", "minimalist", "surreal", "hyper-realistic",
            "abstract", "geometric", "flowing", "chaotic", "symmetrical", "asymmetrical",
            "massive", "tiny", "gigantic", "miniature", "floating", "flying", "submerged"
        ]

        # Random settings
        self.random_settings = [
            "in a forest", "in the mountains", "by the sea", "in space", "underwater",
            "in a desert", "in a cave", "in a temple", "in ruins", "in a futuristic city",
            "in a medieval town", "in a post-apocalyptic wasteland", "in a magical realm",
            "on an alien planet", "in the clouds", "in the depths of the ocean",
            "in the void between worlds", "in a cyberpunk metropolis", "in a steampunk world",
            "in a fantasy kingdom", "in a nightmare realm", "in a dream world",
            "in a parallel universe", "at dawn", "at sunset", "under a full moon",
            "during a storm", "during a battle", "during a celebration", "during a ritual",
            "in the distant future", "in ancient times", "in a forgotten age",
            "in a hidden dimension", "in the afterlife", "in a virtual reality",
            "in a haunted mansion", "in an enchanted garden", "on a battlefield"
        ]

        # Random actions/states
        self.random_actions = [
            "fighting", "exploring", "fleeing", "hunting", "protecting", "destroying",
            "creating", "observing", "transforming", "summoning", "banishing", "healing",
            "corrupting", "building", "tearing down", "rising", "falling", "dancing",
            "sleeping", "awakening", "meditating", "casting a spell", "wielding a weapon",
            "riding", "flying", "swimming", "running", "climbing", "descending",
            "gazing into the distance", "confronting an enemy", "discovering a secret",
            "mourning a loss", "celebrating a victory", "undergoing a transformation",
            "mastering a power", "losing control", "gaining enlightenment", "making a sacrifice"
        ]

    def load_config(self):
        """Load configuration including API keys from config.json"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            return {}

    def setup_ai_models(self):
        """Configure available AI models based on loaded config"""
        # Initialize availability flags
        self.gemini_available = False
        self.openai_available = False
        self.anthropic_available = False
        self.qwen_available = False
        self.ollama_available = False
        
        # Check and configure Gemini
        if GEMINI_AVAILABLE:
            gemini_api_key = self.config.get("GEMINI_API_KEY", "")
            if gemini_api_key and gemini_api_key != "your_gemini_api_key":
                genai.configure(api_key=gemini_api_key, transport='rest')
                self.gemini_available = True
                print("Gemini API configured successfully")
        
        # Check and configure OpenAI
        if OPENAI_AVAILABLE:
            openai_api_key = self.config.get("OPENAI_API_KEY", "")
            if openai_api_key and openai_api_key != "your_openai_api_key":
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                self.openai_available = True
                print("OpenAI API configured successfully")
        
        # Check and configure Anthropic (Claude)
        if ANTHROPIC_AVAILABLE:
            anthropic_api_key = self.config.get("ANTHROPIC_API_KEY", "")
            if anthropic_api_key and anthropic_api_key != "your_claude_api_key":
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                self.anthropic_available = True
                print("Anthropic (Claude) API configured successfully")
        
        # Check and configure Qwen
        if QWEN_AVAILABLE:
            qwen_api_key = self.config.get("QWEN_API_KEY", "")
            if qwen_api_key and qwen_api_key != "your_qwen_api_key":
                dashscope.api_key = qwen_api_key
                self.qwen_available = True
                print("Qwen API configured successfully")
        
        # Check and configure Ollama
        if OLLAMA_AVAILABLE:
            ollama_url = self.config.get("OLLAMA_URL", "")
            if ollama_url:
                self.ollama_url = ollama_url
                self.ollama_available = True
                print("Ollama configured with URL:", ollama_url)
                
        # Overall AI availability
        self.any_ai_available = any([
            self.gemini_available, self.openai_available, 
            self.anthropic_available, self.qwen_available,
            self.ollama_available
        ])
        
        if not self.any_ai_available:
            print("No AI models available - Smart Prompt Generator will work without AI enhancement")

    def get_style_categories(self):
        """Return a list of available style categories from the data directory"""
        categories = {}
        datadir = pathlib.Path(__file__).parent / 'data'

        # Prioritize these popular categories to appear first
        priority_categories = [
            "artist", "general-arts", "movies", "Anime", "lighting",
            "camera", "mood", "digital_artform"
        ]

        # Start with priority categories
        for category in priority_categories:
            if category in self.styler_data.keys():
                categories[category] = list(self.styler_data[category].keys())

        # Add all other categories
        for category in self.styler_data.keys():
            if category not in categories:
                categories[category] = list(self.styler_data[category].keys())

        return categories

    def ensure_random_seed(self, user_seed=0):
        """Ensure we use a new random seed that changes each run"""
        # For true randomness, use a combination of:
        # 1. Current time in nanoseconds
        # 2. A random UUID (which includes hardware addresses and more entropy)
        # 3. The user's seed if provided
        # 4. The run counter which increases on each call

        # Generate a completely random seed
        current_time_ns = time.time_ns()  # Nanosecond precision
        random_uuid = uuid.uuid4().int & 0xFFFFFFFF  # Use last 32 bits of UUID

        # Increment the run counter to ensure different seeds even if called in quick succession
        self.run_counter += 1

        if user_seed > 0:
            # If user provided a seed, use it as a base but still ensure randomness
            final_seed = (user_seed + current_time_ns + random_uuid + self.run_counter) % 2147483647
        else:
            # Pure random seed
            final_seed = (current_time_ns + random_uuid + self.run_counter) % 2147483647

        # Set the seed and return the value used
        random.seed(final_seed)
        return final_seed

    @classmethod
    def INPUT_TYPES(cls):
        # Create an instance to get the style categories
        instance = cls()

        # Create a merged dictionary of all available styles
        style_inputs = {}

        # Add most popular categories as required inputs with multiple choice
        required_categories = ["artist", "general-arts", "movies", "Anime", "mood", "digital_artform"]
        for category in required_categories:
            if category in instance.style_categories:
                style_inputs[f"{category}_style"] = ([
                    "None", *[style for style in instance.style_categories[category] if style != "None"]
                ], {"default": "None"})

        # Create inputs based on API availability
        inputs = {
            "required": {
                "base_prompt": ("STRING", {"default": "A beautiful landscape", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "random_mode": (["Disabled", "Random Styles Only", "Random Base+Styles", "Fully Random"], {"default": "Disabled"}),
                "num_random_styles": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "randomize_seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1, "tooltip": "Seed value used as a base for randomization. A new random seed is generated on every run for variety. The number shown is from the last run."}),
                "preserve_user_text": ("BOOLEAN", {"default": True}),
                **style_inputs
            },
            "optional": {
                "subject_type": (["Person", "Landscape", "Object", "Animal", "Concept", "Scene", "Auto-detect"], {"default": "Auto-detect"}),
                "prompt_template": ("STRING", {"default": "", "multiline": True}),
            }
        }

        # Only add AI enhancement options if any AI model is available
        if instance.any_ai_available:
            # Determine which AI providers are available
            ai_providers = ["None"]
            if instance.gemini_available:
                ai_providers.append("Gemini")
            if instance.openai_available:
                ai_providers.append("OpenAI")
            if instance.anthropic_available:
                ai_providers.append("Claude")
            if instance.qwen_available:
                ai_providers.append("Qwen")
            if instance.ollama_available:
                ai_providers.append("Ollama")
            
            ai_inputs = {
                "enhance_with_ai": ("BOOLEAN", {"default": True}),
                "ai_provider": (ai_providers, {"default": ai_providers[1] if len(ai_providers) > 1 else "None"}),
                "direct_ai_output": ("BOOLEAN", {"default": False, "tooltip": "When enabled, output is exactly what comes from the AI with no additional modifications"}),
            }
            
            # Add Gemini specific options
            if instance.gemini_available:
                ai_inputs["gemini_model"] = (["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {"default": "gemini-1.5-flash"})
            
            # Add OpenAI specific options
            if instance.openai_available:
                ai_inputs["openai_model"] = (["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"})
            
            # Add Claude specific options
            if instance.anthropic_available:
                ai_inputs["claude_model"] = (["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-haiku-20240307"})
            
            # Add Qwen specific options
            if instance.qwen_available:
                ai_inputs["qwen_model"] = (["qwen-plus", "qwen-max", "qwen-max-longcontext"], {"default": "qwen-plus"})
            
            # Add Ollama specific options
            if instance.ollama_available:
                ai_inputs["ollama_model"] = (["llama3", "gemma", "mistral", "phi3", "llava"], {"default": "llama3"})
            
            # Common AI settings
            ai_inputs["creativity_level"] = (["Low", "Medium", "High", "Extreme"], {"default": "Medium"})
            ai_inputs["focus_on"] = (["Realism", "Fantasy", "Abstract", "Artistic", "Photographic", "Cinematic", "Balanced"], {"default": "Balanced"})
            
            inputs["required"].update(ai_inputs)

        return inputs

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt")
    FUNCTION = "generate_smart_prompt"
    CATEGORY = "AI API/Prompt"

    def generate_random_base_prompt(self):
        """Generate a random base prompt idea"""
        # Always use a new random seed when generating random base prompts
        # Random seed is already set in the main function

        # Select random elements for the prompt
        adjective1 = random.choice(self.random_adjectives)
        adjective2 = random.choice(self.random_adjectives)
        subject = random.choice(self.random_subjects)
        setting = random.choice(self.random_settings)
        action = random.choice(self.random_actions)

        # Different prompt structures for variety
        prompt_structures = [
            f"A {adjective1}, {adjective2} {subject} {setting}, {action}",
            f"{adjective1.capitalize()} {subject} {action} {setting}",
            f"The {subject} of {adjective1} dreams, {setting}",
            f"{adjective1.capitalize()} and {adjective2} {subject} {setting}",
            f"A {subject} {action}, {adjective1} and {adjective2}, {setting}",
            f"{setting.capitalize()}, a {adjective1} {subject} is {action}",
            f"The {adjective1} {subject}, {action} {setting}",
            f"A scene of a {adjective1} {subject} {action} {setting}",
            f"{adjective1.capitalize()} {subject} with {adjective2} elements {setting}",
            f"A {adjective1} representation of a {subject} {setting}"
        ]

        # Choose a random structure
        base_prompt = random.choice(prompt_structures)

        return base_prompt

    def select_random_styles(self, num_styles=3):
        """Randomly select styles from different categories"""
        # Random seed is already set in the main function

        random_styles = {}
        # Prioritize using these categories first for random selection
        priority_categories = [
            "artist", "general-arts", "movies", "Anime", "mood", "digital_artform",
            "lighting", "camera", "camera_angles"
        ]

        available_categories = list(self.styler_data.keys())
        # Start with priority categories
        categories_to_use = [cat for cat in priority_categories if cat in available_categories]
        # Add other categories if needed
        if len(categories_to_use) < num_styles:
            remaining = [cat for cat in available_categories if cat not in categories_to_use]
            categories_to_use.extend(remaining[:num_styles - len(categories_to_use)])

        # Shuffle and take the requested number
        random.shuffle(categories_to_use)
        categories_to_use = categories_to_use[:num_styles]

        # For each selected category, pick a random style (excluding "None")
        for category in categories_to_use:
            styles = [s for s in list(self.styler_data[category].keys()) if s != "None"]
            if styles:  # Make sure there are styles available
                selected_style = random.choice(styles)
                random_styles[f"{category}_style"] = selected_style

        return random_styles

    def combine_styles(self, base_prompt, negative_prompt, **kwargs):
        """Combine multiple style elements into a cohesive prompt"""
        # Start with the base prompt
        styled_prompt = base_prompt
        styled_negative = negative_prompt

        # Keep track of applied styles for analysis
        applied_styles = []

        # Process each style selection
        for param_name, style_name in kwargs.items():
            if not param_name.endswith('_style') or style_name == "None":
                continue

            # Extract the category name from the parameter name
            category = param_name.replace('_style', '')

            # Skip if category not found
            if category not in self.styler_data.keys():
                continue

            # Skip if style not found
            if style_name not in self.styler_data[category]:
                continue

            # Apply the style template
            template = self.styler_data[category][style_name]
            styled_prompt, styled_negative = template.replace_prompts(styled_prompt, styled_negative)
            applied_styles.append(f"{category}:{style_name}")

        return styled_prompt, styled_negative, applied_styles

    def enhance_with_gemini(self, prompt, negative_prompt, gemini_model, creativity_level, focus_on, applied_styles, subject_type, preserve_user_text):
        """Use Gemini API to enhance the prompt with creative elements"""
        if not self.gemini_available:
            return prompt, negative_prompt, "API not available"

        try:
            # Configure generation parameters based on creativity level
            temperature_map = {
                "Low": 0.4,
                "Medium": 0.7,
                "High": 0.9,
                "Extreme": 1.0
            }

            generation_config = {
                "temperature": temperature_map.get(creativity_level, 0.7),
                "top_p": 0.9,
                "top_k": 40
            }

            # Common negative terms to filter out from positive prompts
            negative_terms = ["deformed", "blurry", "bad anatomy", "disfigured", "poorly drawn",
                             "ugly", "duplicate", "morbid", "mutilated", "mutation", "deformed",
                             "extra limbs", "ugly", "low quality", "pixelated", "low resolution",
                             "boring", "grainy", "error", "bad", "poor", "poorly", "gross", "horrible",
                             "mutant", "twisted", "disgusting", "unattractive", "bad quality", "worst quality",
                             "extra fingers", "malformed", "malformed hands", "distorted", "distortion", "bad proportions",
                             "misshapen", "cropped", "jpeg artifacts", "watermark", "signature", "text", "logo",
                             "out of frame", "lowres", "worst quality", "bad artist", "amateur", "missing fingers"]

            # Create a system prompt to guide Gemini - STRONGER instructions to never mix positive and negative
            system_prompt = f"""
            You're a creative AI assistant specializing in creating detailed, high-quality image prompts for generative AI.

            APPLIED STYLES: {', '.join(applied_styles)}
            FOCUS: {focus_on}
            SUBJECT TYPE: {subject_type}
            CREATIVITY LEVEL: {creativity_level}
            PRESERVE USER TEXT: {"Yes" if preserve_user_text else "No"}

            Your task is to enhance this prompt by:
            1. Adding more specific, artistic details
            2. Maintaining the original intent and subject matter
            3. Creating harmony between the different styles applied
            4. Being specific about visual elements (lighting, composition, colors, etc.)
            5. Optimizing for {focus_on.lower()} aspects

            STRICT RULES:
            - {"IMPORTANT: The original user text MUST be kept exactly as is at the beginning of the enhanced prompt" if preserve_user_text else "Maintain the style and core elements of the original prompt"}
            - Never remove any key elements from the original prompt
            - Don't add unnecessary parentheses
            - Don't add artist names unless they're already in the original prompt
            - Return ONLY the enhanced prompt text, no explanations or formatting
            - Keep the prompt under 150 words but detailed and descriptive
            - CRITICAL: NEVER include ANY negative elements in the positive prompt
            - CRITICAL: NEVER mention what to avoid in the positive prompt - those go in the negative prompt only
            - NEVER include words like: deformed, blurry, bad anatomy, disfigured, poorly drawn, ugly, pixelated, low quality
            - Focus ONLY on what should be in the image, not what shouldn't be in the image
            """

            # Create the model and generate content
            model = genai.GenerativeModel(gemini_model)

            # Prompt enhancement - TOTALLY SEPARATE from negative prompt
            input_prompt = f"""
            {system_prompt}

            ORIGINAL PROMPT: {prompt}

            Create ONLY a POSITIVE ENHANCED PROMPT with what should be in the image:
            """
            prompt_response = model.generate_content(input_prompt, generation_config=generation_config)
            enhanced_prompt = prompt_response.text.strip()

            # Generate negative prompt as a completely separate call with no reference to the positive prompt
            input_negative = f"""
            You are an AI image generation assistant focused on creating effective negative prompts.

            A negative prompt tells the AI generator what to AVOID, not what to include.

            Create a technical negative prompt listing ONLY elements to avoid:
            - Low quality elements (blurry, pixelated, low resolution)
            - Anatomical problems (deformed limbs, extra fingers, bad hands, ugly)
            - Common AI artifacts (poorly drawn face, asymmetry, distortion)

            The negative prompt should be concise, focused on technical flaws to avoid.

            Return ONLY the negative prompt with NO explanations or additional text.
            DO NOT include ANY positive elements or anything you want to see in the image.
            """
            negative_response = model.generate_content(input_negative, generation_config=generation_config)
            enhanced_negative = negative_response.text.strip()

            # Clean up any extraneous text the model might have added
            if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
                enhanced_prompt = enhanced_prompt[1:-1]

            if enhanced_negative.startswith('"') and enhanced_negative.endswith('"'):
                enhanced_negative = enhanced_negative[1:-1]

            # Additional aggressive filtering to ensure negative terms don't appear in positive prompt
            enhanced_prompt_lower = enhanced_prompt.lower()
            for term in negative_terms:
                if term in enhanced_prompt_lower:
                    # Remove the negative term and any comma before or after it
                    enhanced_prompt_lower = enhanced_prompt_lower.replace(f", {term}", "")
                    enhanced_prompt_lower = enhanced_prompt_lower.replace(f" {term},", "")
                    enhanced_prompt_lower = enhanced_prompt_lower.replace(f" {term} ", " ")
                    enhanced_prompt_lower = enhanced_prompt_lower.replace(f", {term} ", ", ")
                    enhanced_prompt_lower = enhanced_prompt_lower.replace(f" {term}, ", " ")

            # Replace with properly capitalized version
            # Capitalize first letter and after each period
            words = enhanced_prompt_lower.split()
            if words:
                words[0] = words[0].capitalize()

            for i in range(1, len(words)):
                if words[i-1].endswith('.'):
                    words[i] = words[i].capitalize()

            enhanced_prompt = ' '.join(words)

            # Final sanity check for negative terms
            if any(term in enhanced_prompt.lower() for term in negative_terms):
                # If we still have negative terms, regenerate without AI enhancement
                print("Warning: Negative terms detected in enhanced prompt. Using base prompt instead.")
                return prompt, enhanced_negative, f"{gemini_model} (fallback to base prompt)"

            return enhanced_prompt, enhanced_negative, gemini_model

        except Exception as e:
            print(f"Error enhancing prompt with Gemini: {str(e)}")
            return prompt, negative_prompt, f"Error: {str(e)}"

    def generate_fully_random_prompt(self, gemini_model=None):
        """Generate a completely random prompt using Gemini API or fallback to basic random if API unavailable"""
        if not self.gemini_available:
            # Fallback to basic random prompt if no API key
            return self.generate_random_base_prompt(), "", "No API"

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40
            }

            # Create the model
            model = genai.GenerativeModel(gemini_model)

            # Generate a completely random prompt and negative prompt in separate calls
            # First, generate the positive prompt
            positive_system_prompt = """
            Generate a creative, detailed prompt for an AI image generator.

            Your prompt should:
            1. Describe an interesting subject, scene, or concept
            2. Include artistic style, mood, lighting, and composition details
            3. Be specific, detailed, and visually rich
            4. Be suitable for high-quality image generation
            5. Be unexpected and creative

            Return ONLY the prompt text with no explanations, disclaimers, or additional commentary.
            Do NOT include any negative elements in this positive prompt.
            """

            # Generate the positive prompt
            positive_response = model.generate_content(positive_system_prompt, generation_config=generation_config)
            positive = positive_response.text.strip()

            # Now separately generate a negative prompt
            negative_system_prompt = f"""
            Based on this positive prompt for AI image generation:
            "{positive}"

            Generate a suitable NEGATIVE PROMPT listing only elements that should be avoided in the image.

            Focus on technical issues like:
            - Low quality elements (blurry, pixelated)
            - Anatomical problems (deformed limbs, extra fingers)
            - Common AI artifacts to avoid

            Return ONLY the negative prompt with no explanations or additional text.
            """

            negative_response = model.generate_content(negative_system_prompt, generation_config=generation_config)
            negative = negative_response.text.strip()

            # Clean up any extraneous text
            if positive.startswith('"') and positive.endswith('"'):
                positive = positive[1:-1]

            if negative.startswith('"') and negative.endswith('"'):
                negative = negative[1:-1]

            return positive, negative, gemini_model

        except Exception as e:
            print(f"Error generating fully random prompt with Gemini: {str(e)}")
            # Fallback to basic random prompt if API call fails
            return self.generate_random_base_prompt(), "", f"Error: {str(e)}"

    def generate_smart_prompt(self, base_prompt, negative_prompt, random_mode, num_random_styles,
                             randomize_seed, preserve_user_text, subject_type="Auto-detect",
                             prompt_template="", **kwargs):
        """Generate a smart prompt by combining styles and enhancing with AI"""

        # Extract optional AI parameters if they exist
        enhance_with_ai = kwargs.get("enhance_with_ai", False)
        ai_provider = kwargs.get("ai_provider", "None")
        direct_ai_output = kwargs.get("direct_ai_output", False)
        creativity_level = kwargs.get("creativity_level", "Medium")
        focus_on = kwargs.get("focus_on", "Balanced")

        # Check if AI enhancement is available
        ai_available = self.any_ai_available and enhance_with_ai and ai_provider != "None"

        # Use the ensure_random_seed method to generate a new random seed each run
        # This ensures we get different results on each run
        # We pass the randomize_seed parameter but it will still ensure randomness
        used_seed = self.ensure_random_seed(randomize_seed)

        print(f"Using random mode: {random_mode}")
        print(f"Random seed: {used_seed} (automatically changes every run)")
        print(f"AI enhancement: {'Available and enabled' if ai_available else 'Not available or disabled'}")
        print(f"AI provider: {ai_provider}")
        print(f"Direct AI output: {direct_ai_output}")
        print(f"Preserve user text: {preserve_user_text}")

        # Keep track of the original user prompt - save this for later use
        original_prompt = base_prompt
        original_negative = negative_prompt
        used_model = "None"

        # For direct AI output with no styling, just send directly to AI
        if direct_ai_output and ai_available:
            print("Using direct AI output mode - bypassing styling and directly using AI")
            # If in random mode, generate a random base first
            if random_mode == "Fully Random":
                if ai_provider == "Gemini" and self.gemini_available:
                    base_prompt, negative_prompt, _ = self.generate_fully_random_prompt(kwargs.get("gemini_model", "gemini-1.5-flash"))
                else:
                    base_prompt = self.generate_random_base_prompt()
                    negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, disfigured"
                print(f"Generated fully random base prompt for AI: {base_prompt}")
            elif random_mode == "Random Base+Styles":
                base_prompt = self.generate_random_base_prompt()
                print(f"Generated random base prompt for AI: {base_prompt}")
            
            # Now send directly to AI
            enhanced_prompt, enhanced_negative, used_model = self.enhance_with_ai(
                base_prompt, negative_prompt, ai_provider, creativity_level, focus_on,
                ["None"], subject_type, True, **kwargs
            )
            return (enhanced_prompt, enhanced_negative)
            
        # Standard processing flow with styling
        # Handle different random modes
        if random_mode == "Fully Random":
            # Generate a completely random prompt
            if ai_provider == "Gemini" and self.gemini_available:
                base_prompt, negative_prompt, used_model = self.generate_fully_random_prompt(kwargs.get("gemini_model", "gemini-1.5-flash"))
            else:
                base_prompt = self.generate_random_base_prompt()
                negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, disfigured"
                used_model = "Random generator (No API)"

            print(f"Generated fully random base prompt: {base_prompt}")

            # Now apply random styles to this random base
            random_styles = self.select_random_styles(num_random_styles)
            # Apply these styles
            for style_key, style_value in random_styles.items():
                category = style_key.replace('_style', '')
                print(f"Randomly selected: {category} = {style_value}")

            style_kwargs = kwargs.copy()
            style_kwargs.update(random_styles)

        elif random_mode == "Random Base+Styles":
            # Generate a random base prompt
            base_prompt = self.generate_random_base_prompt()
            print(f"Generated random base prompt: {base_prompt}")

            # Apply random styles
            random_styles = self.select_random_styles(num_random_styles)
            for style_key, style_value in random_styles.items():
                category = style_key.replace('_style', '')
                print(f"Randomly selected: {category} = {style_value}")

            style_kwargs = kwargs.copy()
            style_kwargs.update(random_styles)

        elif random_mode == "Random Styles Only":
            # Keep user's base prompt but apply random styles
            print(f"Using user-provided base prompt: {base_prompt}")

            # Apply random styles
            random_styles = self.select_random_styles(num_random_styles)
            for style_key, style_value in random_styles.items():
                category = style_key.replace('_style', '')
                print(f"Randomly selected: {category} = {style_value}")

            style_kwargs = kwargs.copy()
            style_kwargs.update(random_styles)

        else:  # "Disabled" or any other value
            # Use user-provided prompt and manually selected styles
            print(f"Using manual mode with user-provided prompt and styles")
            style_kwargs = kwargs.copy()

        # First combine all selected styles
        styled_prompt, styled_negative, applied_styles = self.combine_styles(
            base_prompt, negative_prompt, **style_kwargs
        )

        # Use the template if provided, otherwise use the styled prompt
        if prompt_template:
            template_obj = Template(prompt_template, "")
            final_prompt, _ = template_obj.replace_prompts(styled_prompt, "")
        else:
            final_prompt = styled_prompt

        print(f"Styles applied: {', '.join(applied_styles)}")
        print(f"Final base prompt: {styled_prompt}")

        # Preserve user text now, before AI enhancement (only if not using direct AI output)
        if not direct_ai_output and preserve_user_text and original_prompt.strip() and random_mode != "Random Styles Only":
            if not final_prompt.lower().startswith(original_prompt.lower()):
                final_prompt = f"{original_prompt}, {final_prompt}"
                print(f"Added user text before AI enhancement: {final_prompt}")

        # Enhance with AI if requested and available
        if ai_available:
            print(f"Enhancing prompt with {ai_provider} (Creativity: {creativity_level}, Focus: {focus_on})")
            final_prompt, final_negative, used_model = self.enhance_with_ai(
                final_prompt, styled_negative, ai_provider, creativity_level, focus_on,
                applied_styles, subject_type, direct_ai_output, **kwargs
            )
            print(f"Enhanced prompt: {final_prompt}")
            print(f"Enhanced negative prompt: {final_negative}")
        else:
            final_negative = styled_negative
            used_model = "No AI enhancement"

        # Final check to ensure user text is still at the beginning (only if not using direct AI output)
        if not direct_ai_output and preserve_user_text and original_prompt.strip() and random_mode != "Random Styles Only":
            if not final_prompt.lower().startswith(original_prompt.lower()):
                final_prompt = f"{original_prompt}, {final_prompt}"
                print(f"Re-added user text after AI enhancement: {final_prompt}")

        # Return only the enhanced prompt and negative prompt
        return (final_prompt, final_negative)

    def enhance_with_ai(self, prompt, negative_prompt, ai_provider, creativity_level, focus_on, applied_styles, subject_type, direct_ai_output, **kwargs):
        """Use selected AI provider to enhance the prompt with creative elements"""
        
        # If direct_ai_output is True, we'll get a direct response from the AI without preserving user text
        preserve_user_text = not direct_ai_output
        
        # Select the appropriate enhancement method based on the AI provider
        if ai_provider == "Gemini" and self.gemini_available:
            return self.enhance_with_gemini(prompt, negative_prompt, kwargs.get("gemini_model", "gemini-1.5-flash"), 
                                           creativity_level, focus_on, applied_styles, subject_type, preserve_user_text)
        
        elif ai_provider == "OpenAI" and self.openai_available:
            return self.enhance_with_openai(prompt, negative_prompt, kwargs.get("openai_model", "gpt-3.5-turbo"), 
                                           creativity_level, focus_on, applied_styles, subject_type, preserve_user_text)
        
        elif ai_provider == "Claude" and self.anthropic_available:
            return self.enhance_with_claude(prompt, negative_prompt, kwargs.get("claude_model", "claude-3-haiku-20240307"), 
                                           creativity_level, focus_on, applied_styles, subject_type, preserve_user_text)
        
        elif ai_provider == "Qwen" and self.qwen_available:
            return self.enhance_with_qwen(prompt, negative_prompt, kwargs.get("qwen_model", "qwen-plus"), 
                                         creativity_level, focus_on, applied_styles, subject_type, preserve_user_text)
        
        elif ai_provider == "Ollama" and self.ollama_available:
            return self.enhance_with_ollama(prompt, negative_prompt, kwargs.get("ollama_model", "llama3"), 
                                           creativity_level, focus_on, applied_styles, subject_type, preserve_user_text)
        
        else:
            print(f"AI provider {ai_provider} not available")
            return prompt, negative_prompt, "AI not available"
            
    def enhance_with_openai(self, prompt, negative_prompt, model, creativity_level, focus_on, applied_styles, subject_type, preserve_user_text):
        """Use OpenAI (ChatGPT) to enhance the prompt"""
        if not self.openai_available:
            return prompt, negative_prompt, "OpenAI not available"
        
        try:
            # Configure temperature based on creativity level
            temperature_map = {
                "Low": 0.4,
                "Medium": 0.7,
                "High": 0.9,
                "Extreme": 1.2
            }
            temperature = temperature_map.get(creativity_level, 0.7)
            
            # Create system prompt for the positive prompt
            system_prompt = f"""
            You're a creative AI assistant specializing in creating detailed, high-quality image prompts for generative AI.

            APPLIED STYLES: {', '.join(applied_styles)}
            FOCUS: {focus_on}
            SUBJECT TYPE: {subject_type}
            CREATIVITY LEVEL: {creativity_level}
            PRESERVE USER TEXT: {"Yes" if preserve_user_text else "No"}

            Your task is to enhance this prompt by:
            1. Adding more specific, artistic details
            2. Maintaining the original intent and subject matter
            3. Creating harmony between the different styles applied
            4. Being specific about visual elements (lighting, composition, colors, etc.)
            5. Optimizing for {focus_on.lower()} aspects

            STRICT RULES:
            - {"IMPORTANT: The original user text MUST be kept exactly as is at the beginning of the enhanced prompt" if preserve_user_text else "Maintain the style and core elements of the original prompt"}
            - Never remove any key elements from the original prompt
            - Don't add unnecessary parentheses
            - Don't add artist names unless they're already in the original prompt
            - Return ONLY the enhanced prompt text, no explanations or formatting
            - Keep the prompt under 150 words but detailed and descriptive
            - CRITICAL: NEVER include ANY negative elements in the positive prompt
            - CRITICAL: NEVER mention what to avoid in the positive prompt - those go in the negative prompt only
            - NEVER include words like: deformed, blurry, bad anatomy, disfigured, poorly drawn, ugly, pixelated, low quality
            - Focus ONLY on what should be in the image, not what shouldn't be in the image
            """
            
            # Enhance positive prompt
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original prompt: {prompt}\n\nCreate ONLY a POSITIVE ENHANCED PROMPT with what should be in the image:"}
                ],
                temperature=temperature
            )
            enhanced_prompt = response.choices[0].message.content.strip()
            
            # Generate negative prompt using a separate call
            negative_system_prompt = """
            You are an AI image generation assistant focused on creating effective negative prompts.

            A negative prompt tells the AI generator what to AVOID, not what to include.

            Create a technical negative prompt listing ONLY elements to avoid:
            - Low quality elements (blurry, pixelated, low resolution)
            - Anatomical problems (deformed limbs, extra fingers, bad hands, ugly)
            - Common AI artifacts (poorly drawn face, asymmetry, distortion)

            The negative prompt should be concise, focused on technical flaws to avoid.

            Return ONLY the negative prompt with NO explanations or additional text.
            DO NOT include ANY positive elements or anything you want to see in the image.
            """
            
            negative_response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": negative_system_prompt},
                    {"role": "user", "content": "Generate a technical negative prompt for image generation:"}
                ],
                temperature=temperature * 0.8  # Slightly lower temperature for negative prompts
            )
            enhanced_negative = negative_response.choices[0].message.content.strip()
            
            return enhanced_prompt, enhanced_negative, model
            
        except Exception as e:
            print(f"Error enhancing prompt with OpenAI: {str(e)}")
            return prompt, negative_prompt, f"OpenAI Error: {str(e)}"
            
    def enhance_with_claude(self, prompt, negative_prompt, model, creativity_level, focus_on, applied_styles, subject_type, preserve_user_text):
        """Use Anthropic Claude to enhance the prompt"""
        if not self.anthropic_available:
            return prompt, negative_prompt, "Claude not available"
        
        try:
            # Configure temperature based on creativity level
            temperature_map = {
                "Low": 0.3,
                "Medium": 0.5,
                "High": 0.7,
                "Extreme": 0.9
            }
            temperature = temperature_map.get(creativity_level, 0.5)
            
            # Create system prompt for the positive prompt
            system_prompt = f"""
            You're a creative AI assistant specializing in creating detailed, high-quality image prompts for generative AI.

            APPLIED STYLES: {', '.join(applied_styles)}
            FOCUS: {focus_on}
            SUBJECT TYPE: {subject_type}
            CREATIVITY LEVEL: {creativity_level}
            PRESERVE USER TEXT: {"Yes" if preserve_user_text else "No"}

            Your task is to enhance this prompt by:
            1. Adding more specific, artistic details
            2. Maintaining the original intent and subject matter
            3. Creating harmony between the different styles applied
            4. Being specific about visual elements (lighting, composition, colors, etc.)
            5. Optimizing for {focus_on.lower()} aspects

            STRICT RULES:
            - {"IMPORTANT: The original user text MUST be kept exactly as is at the beginning of the enhanced prompt" if preserve_user_text else "Maintain the style and core elements of the original prompt"}
            - Never remove any key elements from the original prompt
            - Don't add unnecessary parentheses
            - Don't add artist names unless they're already in the original prompt
            - Return ONLY the enhanced prompt text, no explanations or formatting
            - Keep the prompt under 150 words but detailed and descriptive
            - CRITICAL: NEVER include ANY negative elements in the positive prompt
            - CRITICAL: NEVER mention what to avoid in the positive prompt - those go in the negative prompt only
            - NEVER include words like: deformed, blurry, bad anatomy, disfigured, poorly drawn, ugly, pixelated, low quality
            - Focus ONLY on what should be in the image, not what shouldn't be in the image
            """
            
            # Enhance positive prompt
            positive_msg = f"""
            Original prompt: {prompt}

            Create ONLY a POSITIVE ENHANCED PROMPT with what should be in the image:
            """
            
            response = self.anthropic_client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": positive_msg}]
            )
            enhanced_prompt = response.content[0].text.strip()
            
            # Generate negative prompt using a separate call
            negative_system_prompt = """
            You are an AI image generation assistant focused on creating effective negative prompts.

            A negative prompt tells the AI generator what to AVOID, not what to include.

            Create a technical negative prompt listing ONLY elements to avoid:
            - Low quality elements (blurry, pixelated, low resolution)
            - Anatomical problems (deformed limbs, extra fingers, bad hands, ugly)
            - Common AI artifacts (poorly drawn face, asymmetry, distortion)

            The negative prompt should be concise, focused on technical flaws to avoid.

            Return ONLY the negative prompt with NO explanations or additional text.
            DO NOT include ANY positive elements or anything you want to see in the image.
            """
            
            negative_response = self.anthropic_client.messages.create(
                model=model,
                system=negative_system_prompt,
                max_tokens=500,
                temperature=temperature * 0.8,  # Slightly lower temperature for negative prompts
                messages=[{"role": "user", "content": "Generate a technical negative prompt for image generation:"}]
            )
            enhanced_negative = negative_response.content[0].text.strip()
            
            return enhanced_prompt, enhanced_negative, model
            
        except Exception as e:
            print(f"Error enhancing prompt with Claude: {str(e)}")
            return prompt, negative_prompt, f"Claude Error: {str(e)}"
    
    def enhance_with_ollama(self, prompt, negative_prompt, model, creativity_level, focus_on, applied_styles, subject_type, preserve_user_text):
        """Use Ollama local models to enhance the prompt"""
        if not self.ollama_available:
            return prompt, negative_prompt, "Ollama not available"
        
        try:
            # Configure temperature based on creativity level
            temperature_map = {
                "Low": 0.4,
                "Medium": 0.7,
                "High": 0.9,
                "Extreme": 1.0
            }
            temperature = temperature_map.get(creativity_level, 0.7)
            
            # Create system prompt for the positive prompt
            system_prompt = f"""
            You're a creative AI assistant specializing in creating detailed, high-quality image prompts for generative AI.

            APPLIED STYLES: {', '.join(applied_styles)}
            FOCUS: {focus_on}
            SUBJECT TYPE: {subject_type}
            CREATIVITY LEVEL: {creativity_level}
            PRESERVE USER TEXT: {"Yes" if preserve_user_text else "No"}

            Your task is to enhance this prompt by:
            1. Adding more specific, artistic details
            2. Maintaining the original intent and subject matter
            3. Creating harmony between the different styles applied
            4. Being specific about visual elements (lighting, composition, colors, etc.)
            5. Optimizing for {focus_on.lower()} aspects

            STRICT RULES:
            - {"IMPORTANT: The original user text MUST be kept exactly as is at the beginning of the enhanced prompt" if preserve_user_text else "Maintain the style and core elements of the original prompt"}
            - Never remove any key elements from the original prompt
            - Don't add unnecessary parentheses
            - Don't add artist names unless they're already in the original prompt
            - Return ONLY the enhanced prompt text, no explanations or formatting
            - Keep the prompt under 150 words but detailed and descriptive
            - CRITICAL: NEVER include ANY negative elements in the positive prompt
            - CRITICAL: NEVER mention what to avoid in the positive prompt - those go in the negative prompt only
            - NEVER include words like: deformed, blurry, bad anatomy, disfigured, poorly drawn, ugly, pixelated, low quality
            - Focus ONLY on what should be in the image, not what shouldn't be in the image
            """
            
            # Format the message for Ollama
            user_msg = f"Original prompt: {prompt}\n\nCreate ONLY a POSITIVE ENHANCED PROMPT with what should be in the image:"
            
            # Call Ollama API for positive prompt
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{system_prompt}\n\n{user_msg}",
                    "temperature": temperature,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_prompt = result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            # Generate negative prompt using a separate call
            negative_system_prompt = """
            You are an AI image generation assistant focused on creating effective negative prompts.

            A negative prompt tells the AI generator what to AVOID, not what to include.

            Create a technical negative prompt listing ONLY elements to avoid:
            - Low quality elements (blurry, pixelated, low resolution)
            - Anatomical problems (deformed limbs, extra fingers, bad hands, ugly)
            - Common AI artifacts (poorly drawn face, asymmetry, distortion)

            The negative prompt should be concise, focused on technical flaws to avoid.

            Return ONLY the negative prompt with NO explanations or additional text.
            DO NOT include ANY positive elements or anything you want to see in the image.
            """
            
            negative_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{negative_system_prompt}\n\nGenerate a technical negative prompt for image generation:",
                    "temperature": temperature * 0.8,
                    "stream": False
                }
            )
            
            if negative_response.status_code == 200:
                negative_result = negative_response.json()
                enhanced_negative = negative_result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error for negative prompt: {negative_response.status_code}")
            
            return enhanced_prompt, enhanced_negative, model
            
        except Exception as e:
            print(f"Error enhancing prompt with Ollama: {str(e)}")
            return prompt, negative_prompt, f"Ollama Error: {str(e)}"
            
    def enhance_with_qwen(self, prompt, negative_prompt, model, creativity_level, focus_on, applied_styles, subject_type, preserve_user_text):
        """Use Qwen to enhance the prompt"""
        if not self.qwen_available:
            return prompt, negative_prompt, "Qwen not available"
        
        try:
            # Configure parameters based on creativity level
            temperature_map = {
                "Low": 0.3,
                "Medium": 0.5,
                "High": 0.7,
                "Extreme": 0.9
            }
            temperature = temperature_map.get(creativity_level, 0.5)
            
            # Create system prompt for the positive prompt
            system_prompt = f"""
            You're a creative AI assistant specializing in creating detailed, high-quality image prompts for generative AI.

            APPLIED STYLES: {', '.join(applied_styles)}
            FOCUS: {focus_on}
            SUBJECT TYPE: {subject_type}
            CREATIVITY LEVEL: {creativity_level}
            PRESERVE USER TEXT: {"Yes" if preserve_user_text else "No"}

            Your task is to enhance this prompt by:
            1. Adding more specific, artistic details
            2. Maintaining the original intent and subject matter
            3. Creating harmony between the different styles applied
            4. Being specific about visual elements (lighting, composition, colors, etc.)
            5. Optimizing for {focus_on.lower()} aspects

            STRICT RULES:
            - {"IMPORTANT: The original user text MUST be kept exactly as is at the beginning of the enhanced prompt" if preserve_user_text else "Maintain the style and core elements of the original prompt"}
            - Never remove any key elements from the original prompt
            - Don't add unnecessary parentheses
            - Don't add artist names unless they're already in the original prompt
            - Return ONLY the enhanced prompt text, no explanations or formatting
            - Keep the prompt under 150 words but detailed and descriptive
            - CRITICAL: NEVER include ANY negative elements in the positive prompt
            - CRITICAL: NEVER mention what to avoid in the positive prompt - those go in the negative prompt only
            - NEVER include words like: deformed, blurry, bad anatomy, disfigured, poorly drawn, ugly, pixelated, low quality
            - Focus ONLY on what should be in the image, not what shouldn't be in the image
            """
            
            # Call Qwen API for positive prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original prompt: {prompt}\n\nCreate ONLY a POSITIVE ENHANCED PROMPT with what should be in the image:"}
            ]
            
            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                result_format='message',
                temperature=temperature,
                max_tokens=1024
            )
            
            if response.status_code == 200:
                enhanced_prompt = response['output']['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"Qwen API error: {response.status_code}")
            
            # Generate negative prompt using a separate call
            negative_system_prompt = """
            You are an AI image generation assistant focused on creating effective negative prompts.

            A negative prompt tells the AI generator what to AVOID, not what to include.

            Create a technical negative prompt listing ONLY elements to avoid:
            - Low quality elements (blurry, pixelated, low resolution)
            - Anatomical problems (deformed limbs, extra fingers, bad hands, ugly)
            - Common AI artifacts (poorly drawn face, asymmetry, distortion)

            The negative prompt should be concise, focused on technical flaws to avoid.

            Return ONLY the negative prompt with NO explanations or additional text.
            DO NOT include ANY positive elements or anything you want to see in the image.
            """
            
            negative_messages = [
                {"role": "system", "content": negative_system_prompt},
                {"role": "user", "content": "Generate a technical negative prompt for image generation:"}
            ]
            
            negative_response = dashscope.Generation.call(
                model=model,
                messages=negative_messages,
                result_format='message',
                temperature=temperature * 0.8,
                max_tokens=512
            )
            
            if negative_response.status_code == 200:
                enhanced_negative = negative_response['output']['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"Qwen API error for negative prompt: {negative_response.status_code}")
            
            return enhanced_prompt, enhanced_negative, model
            
        except Exception as e:
            print(f"Error enhancing prompt with Qwen: {str(e)}")
            return prompt, negative_prompt, f"Qwen Error: {str(e)}"

# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "GeminiSmartPromptGenerator": GeminiSmartPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSmartPromptGenerator": "Gemini Smart Prompt Generator"
}