# ComfyUI GeminiOllama Extension

This extension integrates Google's Gemini API, OpenAI (ChatGPT), Anthropic's Claude, Ollama, Qwen, and various image processing tools into ComfyUI, allowing users to leverage these powerful models and features directly within their ComfyUI workflows.

## Features

- Support for multiple AI APIs:
  - Google Gemini
  - OpenAI (ChatGPT)
  - Anthropic Claude
  - Ollama
  - Alibaba Qwen
- Text and image input capabilities
- Streaming option for real-time responses
- FLUX Resolution tools for image sizing
- ComfyUI Styler for advanced styling options
- Raster to Vector (SVG) conversion
- Text splitting and processing
- Easy integration with ComfyUI workflows

## Nodes

### 1. Gemini API

The Gemini API node allows you to interact with Google's Gemini models:

- Text input field for prompts
- Model selection:
  - gemini-1.5-pro-latest
  - gemini-1.5-pro-exp-0801
  - gemini-1.5-flash
- Streaming option for real-time responses

### 2. OpenAI (ChatGPT) API

Integrate with OpenAI's powerful language models:

- Text input field for prompts
- Model selection:
  - gpt-4-turbo
  - gpt-4
  - gpt-3.5-turbo
- Temperature and max tokens settings
- System message configuration
- Streaming support

### 3. Claude API

Access Anthropic's Claude models for advanced language tasks:

- Text input field for prompts
- Model selection:
  - claude-3-opus
  - claude-3-sonnet
  - claude-3-haiku
- Temperature control
- System prompt configuration
- Streaming capability

### 4. Ollama API

Integrate local language models running via Ollama:

- Text input field for prompts
- Dropdown for selecting Ollama models
- Customizable model options

  ### 5. Qwen API

Access Alibaba's Qwen language models:

- Text input field for prompts
- Model selection:
  - qwen-turbo
  - qwen-plus
  - qwen-max
- Temperature control
- Streaming capability

### 6. FLUX Resolutions

[Previous FLUX Resolutions content remains the same]

### 7. ComfyUI Styler

[Previous ComfyUI Styler content remains the same]

### 8. Raster to Vector (SVG) and Save SVG

[Previous Raster to Vector content remains the same]

### 9. TextSplitByDelimiter

[Previous TextSplitByDelimiter content remains the same]

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/GeminiOllama.git
   ```

2. Install the required dependencies:
   ```
   pip install google-generativeai openai anthropic requests vtracer
   ```

## Configuration

### API Key Setup

1. Create a `config.json` file in the extension directory with the following content:
   ```json
   {
     "GEMINI_API_KEY": "your_gemini_api_key_here",
     "OPENAI_API_KEY": "your_openai_api_key_here",
     "ANTHROPIC_API_KEY": "your_claude_api_key_here",
     "OLLAMA_URL": "http://localhost:11434",
     "QWEN_API_KEY": "your_qwen_api_key_here"
   }
   ```

2. Obtain API keys from:
   - Gemini: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - OpenAI: [OpenAI Platform](https://platform.openai.com/api-keys)
   - Claude: [Anthropic Console](https://console.anthropic.com/)
   - Qwen: [DashScope Console](https://dashscope.console.aliyun.com/)

## Usage

After installation and configuration, new nodes for each API will be available in ComfyUI.

### Input Parameters

- `api_choice`: Choose between "Gemini", "OpenAI", "Claude", and "Ollama"
- `prompt`: The text prompt for the AI model
- `model_selection`: Select the specific model for chosen API
- `temperature`: Control response randomness (OpenAI and Claude)
- `system_message`: Set system behavior (OpenAI and Claude)
- `stream`: Enable/disable streaming responses
- `image` (optional): Input image for vision-based tasks

### Output

- `text`: The generated response from the chosen AI model

## Main Functions

1. `get_api_keys()`: Retrieves API keys from the config file
2. `get_ollama_url()`: Gets the Ollama URL from the config file
3. `generate_content()`: Main function to generate content based on the chosen API and parameters
4. `generate_gemini_content()`: Handles content generation for Gemini API
5. `generate_openai_content()`: Manages content generation for OpenAI API
6. `generate_claude_content()`: Handles content generation for Claude API
7. `generate_ollama_content()`: Manages content generation for Ollama API
8. `tensor_to_image()`: Converts a tensor to a PIL Image for vision-based tasks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
