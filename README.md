
# ComfyUI GeminiOllama Extension

This extension integrates Google's Gemini API, Ollama, and various image processing tools into ComfyUI, allowing users to leverage these powerful models and features directly within their ComfyUI workflows.

## Features

- Support for Gemini and Ollama APIs
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

### 2. Ollama API

Integrate local language models running via Ollama:

- Text input field for prompts
- Dropdown for selecting Ollama models
- Customizable model options

### 3. FLUX Resolutions

Provides advanced image resolution and sizing options:

- Predefined resolution presets (e.g., 768x1024, 1024x768, 1152x768)
- Custom sizing parameters:
  - size_selected
  - multiply_factor
  - manual_width
  - manual_height

### 4. ComfyUI Styler

Extensive styling options for various creative needs:

- Advertising styles (e.g., automotive, corporate, fashion editorial)
- Art styles (e.g., abstract, art deco, cubist, impressionist)
- Futuristic styles (e.g., biomechanical, cyberpunk)
- Additional categories like composition, environment, and texture

### 5. Raster to Vector (SVG) and Save SVG

Convert raster images to vector graphics and save them:

**Raster to Vector node parameters:**

- colormode
- filter_speckle
- corner_threshold
- ... (and more)

**Save SVG node options:**

- filename_prefix
- overwrite_existing

### 6. TextSplitByDelimiter

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
   pip install google-generativeai requests vtracer
   ```

## Configuration

### Gemini API Key Setup

1. Go to the [Google AI Studio](https://makersuite.google.com/app/apikey).
2. Create a new API key or use an existing one.
3. Copy the API key.
4. Create a `config.json` file in the extension directory with the following content:
   ```json
   {
     "GEMINI_API_KEY": "your_api_key_here"
   }
   ```

### Ollama Setup

1. Install Ollama by following the instructions on the [Ollama GitHub page](https://github.com/ollama/ollama).
2. Start the Ollama server (usually runs on `http://localhost:11434`).
3. Add the Ollama URL to your `config.json`:
   ```json
   {
     "GEMINI_API_KEY": "your_api_key_here",
     "OLLAMA_URL": "http://localhost:11434"
   }
   ```

## Usage

After installation and configuration, a new node called "Gemini Ollama API" will be available in ComfyUI.

### Input Parameters

- `api_choice`: Choose between "Gemini" and "Ollama"
- `prompt`: The text prompt for the AI model
- `gemini_model`: Select the Gemini model (for Gemini API)
- `ollama_model`: Specify the Ollama model (for Ollama API)
- `stream`: Enable/disable streaming responses
- `image` (optional): Input image for vision-based tasks

### Output

- `text`: The generated response from the chosen AI model

## Main Functions

1. `get_gemini_api_key()`: Retrieves the Gemini API key from the config file.
2. `get_ollama_url()`: Gets the Ollama URL from the config file.
3. `generate_content()`: Main function to generate content based on the chosen API and parameters.
4. `generate_gemini_content()`: Handles content generation for Gemini API.
5. `generate_ollama_content()`: Manages content generation for Ollama API.
6. `tensor_to_image()`: Converts a tensor to a PIL Image for vision-based tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
