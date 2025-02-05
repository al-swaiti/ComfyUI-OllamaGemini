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

Provides advanced image resolution and sizing options:

- Predefined resolution presets (e.g., 768x1024, 1024x768, 1152x768)
- Custom sizing parameters:
  - size_selected
  - multiply_factor
  - manual_width
  - manual_height

### 7. ComfyUI Styler

Extensive styling options for various creative needs:

🎨 General Arts – A broad spectrum of traditional and modern art styles
🌸 Anime – Bring your designs to life with anime-inspired aesthetics
🎨 Artist – Channel the influence of world-class artists
📷 Camera – Fine-tune focal lengths, angles, and setups
📐 Camera Angles – Add dynamic perspectives with a range of angles
🌟 Aesthetic – Define unique artistic vibes and styles
🎞️ Color Grading – Achieve rich cinematic tones and palettes
🎬 Movies – Get inspired by different cinematic worlds
🖌️ Digital Artform – From vector art to abstract digital styles
💪 Body Type – Customize different body shapes and dimensions
😲 Reactions – Capture authentic emotional expressions
💭 Feelings – Set the emotional tone for each creation
📸 Photographers – Infuse the style of renowned photographers
💇 Hair Style – Wide variety of hair designs for your characters
🏛️ Architecture Style – Classical to modern architectural themes
🛠️ Architect – Designs inspired by notable architects
🚗 Vehicle – Add cars, planes, or futuristic transportation
🕺 Poses – Customize dynamic body positions
🔬 Science – Add futuristic, scientific elements
👗 Clothing State – Define the wear and tear of clothing
👠 Clothing Style – Wide range of fashion styles
🎨 Composition – Control the layout and arrangement of elements
📏 Depth – Add dimensionality and focus to your scenes
🌍 Environment – From nature to urban settings, create rich backdrops
😊 Face – Customize facial expressions and emotions
🦄 Fantasy – Bring magical and surreal elements into your visuals
🎃 Filter – Apply unique visual filters for artistic effects
🖤 Gothic – Channel dark, mysterious, and dramatic themes
👻 Halloween – Get spooky with Halloween-inspired designs
✏️ Line Art – Incorporate clean, bold lines into your creations
💡 Lighting – Set the mood with dramatic lighting effects
✈️ Milehigh – Capture the essence of aviation and travel
🎭 Mood – Set the emotional tone and atmosphere
🎞️ Movie Poster – Create dramatic, story-driven poster designs
🎸 Punk – Channel bold, rebellious aesthetics
🌍 Travel Poster – Design vintage travel posters with global vibes

### 8. Raster to Vector (SVG) and Save SVG

Convert raster images to vector graphics and save them:

**Raster to Vector node parameters:**

- colormode
- filter_speckle
- corner_threshold
- ... (and more)

**Save SVG node options:**

- filename_prefix
- overwrite_existing

### 9. TextSplitByDelimiter

Split text based on specified delimiters:

- Input text field
- Delimiter options:
  - split_regex
  - split_every
  - split_count


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
