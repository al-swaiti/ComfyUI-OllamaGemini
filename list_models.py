import os
import json
import requests
import socket
import httpx
from google import genai
from openai import OpenAI

# Set a global timeout for faster fallback when offline
NETWORK_TIMEOUT = 3  # seconds

def is_network_available():
    """Quick check if network is available - returns within 1 second"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        return True
    except OSError:
        return False

def get_api_keys():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        gemini_api_key = config.get("GEMINI_API_KEY", "")
        openai_api_key = config.get("OPENAI_API_KEY", "")
        return gemini_api_key, openai_api_key
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")
        return None, None

def get_gemini_models():
    # Default models if API call fails
    default_gemini_models = [
        # Gemini 3.x Models (newest)
        "gemini-3-pro-preview",
        "gemini-3-pro-image-preview",
        "gemini-3-flash-preview",
        # Gemini 2.5 Models
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.5-flash",
        "gemini-2.5-flash-image-preview",
        # Gemini 2.0 Models
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-exp-image-generation",
        # Gemini 1.5 Models
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        # Image Generation Models
        "imagen-3.0-generate-002",
        "imagen-4.0-generate-001",
    ]

    # Quick network check first
    if not is_network_available():
        print("[OllamaGemini] Offline mode - using default Gemini models")
        return default_gemini_models

    gemini_api_key, _ = get_api_keys()
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print("Gemini API key is missing. Using default model list.")
        return default_gemini_models

    try:
        # Create client with api_key
        client = genai.Client(api_key=gemini_api_key)
        
        # Use a timeout wrapper
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Wrap client.models.list in a lambda or function
            def fetch_models():
                return list(client.models.list())
            
            future = executor.submit(fetch_models)
            try:
                models = future.result(timeout=NETWORK_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(f"[OllamaGemini] Gemini API timeout ({NETWORK_TIMEOUT}s) - using defaults")
                return default_gemini_models
        
        model_names = [model.name for model in models]
        # Extract just the model name from the full path
        model_names = [name.split('/')[-1] for name in model_names]
        print(f"Gemini models fetched: {len(model_names)} models available")

        # Include common and image generation models that might not be returned by API
        for model in default_gemini_models:
            if model not in model_names:
                model_names.append(model)
        return sorted(model_names)
    except Exception as e:
        print(f"Error fetching Gemini models: {str(e)}")
        print("Using default model list.")
        return default_gemini_models

def get_openai_models():
    # Default models if API call fails
    default_openai_models = [
        # GPT-4 Family
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        # GPT-3.5 Family
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k",
        # O1 Family
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        # DeepSeek Models
        "deepseek-ai/deepseek-r1"
    ]

    # Quick network check first
    if not is_network_available():
        print("[OllamaGemini] Offline mode - using default OpenAI models")
        return default_openai_models

    _, openai_api_key = get_api_keys()
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("OpenAI API key is missing. Using default model list.")
        return default_openai_models

    try:
        # Create client with timeout
        client = OpenAI(
            api_key=openai_api_key,
            timeout=httpx.Timeout(NETWORK_TIMEOUT, connect=2.0)
        )
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        print(f"OpenAI models fetched: {len(model_ids)} models available")

        # Include common models that might not be returned by API
        for model in default_openai_models:
            if model not in model_ids:
                model_ids.append(model)
        return sorted(model_ids)
    except Exception as e:
        print(f"Error fetching OpenAI models: {str(e)}")
        print("Using default OpenAI model list.")
        return default_openai_models

def get_gemini_image_models():
    """
    Dynamically fetches a list of Gemini models that support image generation.
    
    Models:
    - gemini-3-pro-image-preview (Nano Banana Pro): Professional asset production, 4K, 14 reference images
    - gemini-2.5-flash-image-preview (Nano Banana): Fast, efficient, 1024px
    - imagen-3.0-generate-002: High quality image generation
    - imagen-4.0-generate-001: Specialized image generation
    """
    fallback_models = ["gemini-3-pro-image-preview", "gemini-2.5-flash-image-preview", "imagen-4.0-generate-001"]

    # Quick network check first
    if not is_network_available():
        print("[OllamaGemini] Offline mode - using default image models")
        return fallback_models

    gemini_api_key, _ = get_api_keys()
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print("Gemini API key is missing. Using default image model list.")
        return fallback_models

    try:
        # Create client with api_key
        client = genai.Client(api_key=gemini_api_key)
        
        # Use a timeout wrapper
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            def fetch_models():
                return list(client.models.list())
                
            future = executor.submit(fetch_models)
            try:
                all_models = future.result(timeout=NETWORK_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(f"[OllamaGemini] Gemini API timeout ({NETWORK_TIMEOUT}s) - using default image models")
                return fallback_models
        
        image_models = []

        # Simplified logic: Trust models with 'image' in the name, as the API
        # is not returning the full list of supported methods easily.
        for model in all_models:
            # model.name in new API might be 'models/...' or just '...'
            model_name = model.name.split('/')[-1]
            if 'image' in model_name:
                image_models.append(model_name)
        
        # Ensure fallback models are always available
        for model in fallback_models:
            if model not in image_models:
                image_models.append(model)

        if not image_models:
            print("Could not dynamically find any image models. Using a default list.")
            return fallback_models

        print(f"Final Gemini image models list: {image_models}")
        return sorted(list(set(image_models)))
    except Exception as e:
        print(f"Error fetching Gemini image models: {str(e)}")
        print("Using default image model list.")
        return fallback_models


if __name__ == "__main__":
    print("\n=== Gemini Models ===")
    gemini_models = get_gemini_models()
    for model in gemini_models:
        print(f"Name: {model}")
        print("")

    print("\n=== OpenAI Models ===")
    openai_models = get_openai_models()
    for model in openai_models:
        print(f"ID: {model}")
        print("")
