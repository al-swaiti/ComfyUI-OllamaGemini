# ComfyUI-OllamaGemini Examples

This folder contains example images, videos, and workflows for the ComfyUI-OllamaGemini extension.

## Downloading Examples

If you cloned the repository without the examples (using a shallow clone or if the examples were excluded), you can download them separately:

### Option 1: Download from GitHub Web Interface

1. Go to the [examples folder](https://github.com/al-swaiti/ComfyUI-OllamaGemini/tree/main/examples) in the GitHub repository
2. Download individual files by clicking on them and then clicking the "Download" button

### Option 2: Use Git to Fetch Only the Examples Folder

```bash
# If you already cloned the repository without examples
cd ComfyUI-OllamaGemini
git fetch
git checkout origin/main -- examples

# Or to download just the examples folder to a new location
git clone --depth 1 --filter=blob:none --sparse https://github.com/al-swaiti/ComfyUI-OllamaGemini.git
cd ComfyUI-OllamaGemini
git sparse-checkout set examples
```

## Example Files

- **8.mp4**: Introduction video showing the extension's capabilities
- **1.png, 3.png, 4.png, 5.png**: Screenshots of various features
- **6.png**: Background removal demonstration
- **7.png**: ComfyUI Styler demonstration
- **9.gif**: SVG conversion demonstration
- **smart_prompt.png**: Smart Prompt Generator interface
- **smart_prompt_workflow.png**: Example workflow using the Smart Prompt Generator
