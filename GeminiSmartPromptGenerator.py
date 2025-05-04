import os
import json
import random
import time
import pathlib
import uuid
from collections import defaultdict
import importlib.util
from PIL import Image

from .prompt_stylerx import StylerData, Template

class GeminiSmartPromptGenerator:
    """
    A node that intelligently combines multiple style elements from different categories
    to create creative prompts.
    """
    def __init__(self):
        # Load all style data
        self.styler_data = StylerData()
        
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

        # Create inputs
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
                "prompt_template": ("STRING", {"default": "", "multiline": True}),
            }
        }

        return inputs

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt")
    FUNCTION = "generate_smart_prompt"
    CATEGORY = "Prompt"

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

    def generate_smart_prompt(self, base_prompt, negative_prompt, random_mode, num_random_styles,
                             randomize_seed, preserve_user_text, prompt_template="", **kwargs):
        """Generate a smart prompt by combining styles"""

        # Use the ensure_random_seed method to generate a new random seed each run
        # This ensures we get different results on each run
        # We pass the randomize_seed parameter but it will still ensure randomness
        used_seed = self.ensure_random_seed(randomize_seed)

        print(f"Using random mode: {random_mode}")
        print(f"Random seed: {used_seed} (automatically changes every run)")
        print(f"Preserve user text: {preserve_user_text}")

        # Keep track of the original user prompt - save this for later use
        original_prompt = base_prompt
        original_negative = negative_prompt
        
        # Standard processing flow with styling
        # Handle different random modes
        if random_mode == "Fully Random":
            # Generate a completely random prompt
            base_prompt = self.generate_random_base_prompt()
            negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, disfigured"

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
        print(f"Final prompt: {styled_prompt}")

        # Preserve user text if needed
        if preserve_user_text and original_prompt.strip() and random_mode != "Random Styles Only":
            if not final_prompt.lower().startswith(original_prompt.lower()):
                final_prompt = f"{original_prompt}, {final_prompt}"
                print(f"Added user text to final prompt: {final_prompt}")

        # Return the final prompt and negative prompt
        return (final_prompt, styled_negative)

# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "GeminiSmartPromptGenerator": GeminiSmartPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSmartPromptGenerator": "Gemini Smart Prompt Generator"
}