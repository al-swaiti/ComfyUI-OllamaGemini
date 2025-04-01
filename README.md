<div align="center">

# 🚀 ComfyUI GeminiOllama Extension

**Supercharge your ComfyUI workflows with AI superpowers**

[![GitHub stars](https://img.shields.io/github/stars/al-swaiti/ComfyUI-OllamaGemini?style=social)](https://github.com/al-swaiti/ComfyUI-OllamaGemini/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

This extension integrates Google's Gemini API, OpenAI (ChatGPT), Anthropic's Claude, Ollama, Qwen, and various image processing tools into ComfyUI, allowing users to leverage these powerful models and features directly within their ComfyUI workflows.

## Features

<div align="center">

### 1️⃣ Multiple AI API Integrations

<img src="examples/1.png" width="700">

- **Google Gemini**: Access gemini-2.0-pro, gemini-2.0-flash, gemini-1.5-pro and more
- **OpenAI**: Use gpt-4o, gpt-4-turbo, gpt-3.5-turbo, and DeepSeek models
- **Anthropic Claude**: Leverage claude-3.7-sonnet, claude-3.5-sonnet, claude-3-opus and more
- **Alibaba Qwen**: Access qwen-max, qwen-plus, qwen-turbo models
- **Ollama**: Run local models with customizable parameters

### 2️⃣ Gemini Image Generation

<img src="examples/3.png" width="700">
<img src="examples/4.png" width="700">
<img src="examples/5.png" width="700">

- Generate images directly with Google's Gemini 2.0 Flash model
- Customize with prompts and negative prompts
- Automatic saving to ComfyUI's output directory

### 3️⃣ Prompt Enhancement

<img src="examples/1.png" width="700">

- Transform simple prompts into detailed, model-specific instructions
- Multiple specialized templates (SDXL, Wan2.1, FLUX.1-dev, HunyuanVideo)
- Returns only the enhanced prompt without additional commentary

### 4️⃣ Background Removal (BRIA RMBG)

<img src="examples/6.png" width="700">

- High-quality background removal with fine detail preservation
- Preserves complex edges, hair, thin stems, and transparent elements
- Generates both transparent images and alpha masks

### 5️⃣ SVG Conversion

<img src="examples/9.gif" width="700">

- Convert raster images to high-quality vector graphics
- Multiple vectorization parameters for precise control
- Save and preview SVG files directly in ComfyUI

### 6️⃣ FLUX Resolutions

<img src="examples/8.png" width="700">

- Precise image sizing with predefined and custom options
- Multiple resolution presets for various use cases
- Custom sizing parameters for complete control

### 7️⃣ ComfyUI Styler

<img src="examples/7.png" width="700">

- Hundreds of artistic styles for creative control
- Categories include art styles, camera settings, moods, and more
- Easily combine multiple style elements

</div>



## 💻 Installation & Setup

<details open>
<summary><b>📦 Installation</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you don't have it already
2. In ComfyUI, go to the Manager tab and search for "OllamaGemini"
3. Click Install

### Method 2: Manual Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/al-swaiti/ComfyUI-OllamaGemini.git
   ```

2. Install the required dependencies:
   ```bash
   pip install google-generativeai>=0.3.0 openai>=1.3.0 anthropic>=0.8.0 requests>=2.31.0 vtracer>=0.6.0  dashscope>=1.13.6  Pillow>=10.0.0 scipy>=1.10.0  opencv-python transformers>=4.30.0 
   ```

3. Restart ComfyUI
</details>

<details open>
<summary><b>🔑 API Key Setup</b></summary>
### Obtaining API Keys

<table>
<tr>
  <th>Provider</th>
  <th>Where to Get</th>
  <th>Free Tier</th>
</tr>
<tr>
  <td>Google Gemini</td>
  <td><a href="https://makersuite.google.com/app/apikey">Google AI Studio</a></td>
  <td>✅ Yes</td>
</tr>
<tr>
  <td>OpenAI</td>
  <td><a href="https://platform.openai.com/api-keys">OpenAI Platform</a></td>
  <td>❌ No</td>
</tr>
<tr>
  <td>Anthropic Claude</td>
  <td><a href="https://console.anthropic.com/">Anthropic Console</a></td>
  <td>✅ Limited</td>
</tr>
<tr>
  <td>Ollama</td>
  <td><a href="https://ollama.com/">Ollama</a> (runs locally)</td>
  <td>✅ Yes</td>
</tr>
<tr>
  <td>Alibaba Qwen</td>
  <td><a href="[https://dashscope.console.aliyun.com/](https://www.alibabacloud.com/help/en/model-studio/developer-reference/get-api-key)">DashScope Console</a></td>
  <td>✅ Limited</td>
</tr>
</table>
</details>

### Option 1: Using the Config File

Create or edit `config.json` in the extension directory:

```json
{
  "GEMINI_API_KEY": "your_gemini_api_key",
  "OPENAI_API_KEY": "your_openai_api_key",
  "ANTHROPIC_API_KEY": "your_claude_api_key",
  "OLLAMA_URL": "http://localhost:11434",
  "QWEN_API_KEY": "your_qwen_api_key"
}
```


## 🔹 Quick Start Guide

<details open>
<summary><b>💬 Using AI API Services</b></summary>

1. Add the appropriate API node to your workflow (Gemini API, OpenAI API, Claude API, etc.)
2. Enter your prompt in the text field
3. Select the desired model from the dropdown
4. Adjust parameters like temperature and max tokens as needed
5. For enhanced prompts, enable "structure_output" and select a prompt structure template
6. Connect the output to other nodes in your workflow

<img src="examples/1.png" width="500">
</details>

<details>
<summary><b>🖼️ Generating Images with Gemini</b></summary>

1. Add the "Gemini Image Generator" node to your workflow
2. Enter your prompt describing the desired image
3. Optionally add a negative prompt to exclude unwanted elements
4. Connect the output to a preview node to see the generated image

<img src="examples/3.png" width="500">
</details>

<details>
<summary><b>🪄 Removing Backgrounds</b></summary>

1. Add the "BRIA RMBG" node to your workflow
2. Connect an image source to the input
3. Set model_version to 2.0 for best results
4. Connect the image output to see the transparent result
5. Connect the mask output to see the generated mask

<img src="examples/6.png" width="500">
</details>

<details>
<summary><b>✒️ Converting Images to SVG</b></summary>

1. Add the "Convert Image to SVG" node to your workflow
2. Connect an image source to the input
3. Configure the vectorization parameters
4. Connect the output to the "Save SVG File" node
5. Set a filename prefix and enable preview

<img src="examples/9.gif" width="500">
</details>

## 🌟 Why Choose This Extension?

### Comprehensive API Integration

Access the most powerful AI models through a single interface:

- **Google Gemini**: gemini-2.0-pro, gemini-2.0-flash, gemini-1.5-pro, and more
- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, and DeepSeek models
- **Anthropic Claude**: claude-3.7-sonnet, claude-3.5-sonnet, claude-3-opus, and more
- **Alibaba Qwen**: qwen-max, qwen-plus, qwen-turbo, qwen-max-longcontext
- **Ollama**: Run any local model with customizable parameters

### Advanced Prompt Enhancement

Transform simple prompts into detailed, model-specific instructions with specialized templates:

- **SDXL**: Optimized for Stable Diffusion XL with detailed artistic parameters
- **Wan2.1**: Specialized format with subject, setting, and style elements
- **FLUX.1-dev**: Enhanced format with depth effects and camera details
- **HunyuanVideo**: Specialized for video generation with cohesive descriptions
- **Custom**: Create your own prompt structure for specific needs

### High-Quality Tools

- **BRIA RMBG**: Best-in-class background removal with fine detail preservation
- **SVG Conversion**: High-quality vectorization with vtracer
- **FLUX Resolutions**: Precise image sizing with predefined and custom options
- **ComfyUI Styler**: Hundreds of artistic styles for creative control

## 👨‍💻 Contributing

Contributions are welcome! Here's how you can help:

- **Bug Reports**: Open an issue describing the bug and how to reproduce it
- **Feature Requests**: Suggest new features or improvements
- **Pull Requests**: Submit PRs for bug fixes or new features
- **Documentation**: Help improve or translate the documentation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

### ⭐ If you find this extension useful, please consider giving it a star! ⭐

</div>

