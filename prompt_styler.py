import json
import pathlib
from collections import defaultdict

class Template:
    def __init__(self, prompt, negative_prompt, **kwargs):
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def replace_prompts(self, positive_prompt, negative_prompt):
        positive_result = self.prompt.replace('{prompt}', positive_prompt)
        negative_result = ', '.join(x for x in (self.negative_prompt, negative_prompt) if x)
        return positive_result, negative_result

class StylerData:
    def __init__(self, datadir=None):
        self._data = defaultdict(dict)
        if datadir is None:
            datadir = pathlib.Path(__file__).parent / 'data'

        for j in datadir.glob('*/*.json'):
            try:
                with j.open('r') as f:
                    content = json.load(f)
                    group = j.parent.name
                    for template in content:
                        self._data[group][template['name']] = Template(**template)
            except PermissionError:
                print(f"Warning: No read permissions for file {j}")
            except KeyError:
                print(f"Warning: Malformed data in {j}")

    def __getitem__(self, item):
        return self._data[item]

    def keys(self):
        return self._data.keys()

styler_data = StylerData()

class PromptStyler:
    menus = ()

    @classmethod
    def INPUT_TYPES(cls):
        menus = {menu: (list(styler_data[menu].keys()), ) for menu in cls.menus}

        inputs = {
            "required": {
                "text_positive": ("STRING", {"default": "", "multiline": True}),
                "text_negative": ("STRING", {"default": "", "multiline": True}),
                **menus
            },
        }

        return inputs

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('text_positive','text_negative',)
    FUNCTION = 'prompt_styler'
    CATEGORY = "AI API"

    def prompt_styler(self, text_positive, text_negative, log_prompt, **kwargs):
        text_positive_styled, text_negative_styled = text_positive, text_negative
        for menu, selection in kwargs.items():
            text_positive_styled, text_negative_styled = styler_data[menu][selection].replace_prompts(text_positive_styled, text_negative_styled)
 


NODES = {
    'ComfyUI Styler': (
        'general-arts','Aesthetic', 'Anime', 'artist', 'camera', 'camera_angles',
        'clothing_state', 'clothing_style', 'composition', 'depth', 'environment', 'face', 'Fantasy',
        'filter', 'Gothic', 'Halloween','Line_Art' ,'lighting', 'milehigh', 'mood', 'Movie_Poster', 'Punk', 'Travel_Poster'
    ),
}

