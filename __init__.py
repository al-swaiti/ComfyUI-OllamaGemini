import os
import sys
import filecmp
import shutil
import __main__
import json

python = sys.executable

extensions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                 "web" + os.sep + "extensions" + os.sep + "GeminiOllama")
javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

if not os.path.exists(extensions_folder):
    print('Making the "web\extensions\GeminiOllama" folder')
    os.mkdir(extensions_folder)

result = filecmp.dircmp(javascript_folder, extensions_folder)

if result.left_only or result.diff_files:
    print('Update to javascript files detected')
    file_list = list(result.left_only)
    file_list.extend(x for x in result.diff_files if x not in file_list)

    for file in file_list:
        print(f'Copying {file} to extensions folder')
        src_file = os.path.join(javascript_folder, file)
        dst_file = os.path.join(extensions_folder, file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copy(src_file, dst_file)

# create config
if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")):
    config = {
        "GEMINI_API_KEY": "your key",
        "OLLAMA_URL": "http://localhost:11434"
    }
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "w") as f:
        json.dump(config, f, indent=4)

# load config
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r") as f:
    config = json.load(f)

from .GeminiOllamaNode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
