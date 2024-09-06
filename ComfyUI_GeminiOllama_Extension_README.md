
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

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/GeminiOllama.git
   ```

2. Install the required dependencies:

   ```bash
   pip install google-generativeai vtracer
   ```

## Configuration

### Gemini API Key Setup

1. Obtain a Gemini API key from the Google AI Studio.
2. Create a `config.json` file in the extension directory:

   ```json
   {
     "GEMINI_API_KEY": "your_api_key_here"
   }
   ```

### Ollama Setup

1. Install Ollama following the instructions on the Ollama GitHub page.
2. Start the Ollama server (usually runs on http://localhost:11434).
3. (Optional) Add the Ollama URL to your `config.json`:

   ```json
   {
     "GEMINI_API_KEY": "your_api_key_here",
     "OLLAMA_URL": "http://localhost:11434"
   }
   ```

## Usage

After installation and configuration, the new nodes will be available in ComfyUI. Drag and drop them into your workflow to start using their features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
