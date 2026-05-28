import os
import json
import time
import threading
import requests
import socket
import httpx
from google import genai
from openai import OpenAI

# Set a global timeout for faster fallback when offline
NETWORK_TIMEOUT = 3  # seconds

# ComfyUI invokes INPUT_TYPES() on every prompt validation, so each node that
# calls one of the get_*_models() helpers below would otherwise hit the
# provider's /models endpoint on every queued prompt. Under concurrent load
# that exhausts Google's per-project quota (HTTP 429 RESOURCE_EXHAUSTED) and
# the helper falls back to a hard-coded list — which may not include the
# model the user's workflow is pinned to, causing the validator to silently
# strip the node. A short module-level TTL cache eliminates the burst.
# Only successful live responses are cached; failures return the fallback
# uncached so the next call retries.
#
# Single-flight: a per-provider lock serializes concurrent cache misses so
# only one thread actually hits the live API while the rest wait and reuse
# the populated value. Without this, every cold start and every TTL boundary
# would still produce a synchronous burst from all concurrent validators.
_MODEL_LIST_TTL = 300  # seconds
_model_list_cache = {}
_model_list_lock = threading.Lock()
_fetch_locks = {}
_fetch_locks_meta = threading.Lock()
_last_config_mtime = 0


def _check_config_changed():
    global _last_config_mtime
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    try:
        mtime = os.path.getmtime(config_path)
        # Initialize on first run so we don't clear an empty cache unnecessarily
        if _last_config_mtime == 0:
            _last_config_mtime = mtime
            return False
        if mtime > _last_config_mtime:
            _last_config_mtime = mtime
            return True
    except OSError:
        pass
    return False


def _get_cached(key):
    with _model_list_lock:
        if _check_config_changed():
            _model_list_cache.clear()
            
        entry = _model_list_cache.get(key)
        if entry and time.monotonic() < entry["expires_at"]:
            value = entry["value"]
            return list(value) if isinstance(value, list) else value
    return None


def _set_cached(key, value):
    with _model_list_lock:
        cached_value = list(value) if isinstance(value, list) else value
        _model_list_cache[key] = {"value": cached_value, "expires_at": time.monotonic() + _MODEL_LIST_TTL}


def _fetch_lock_for(key):
    with _fetch_locks_meta:
        lock = _fetch_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _fetch_locks[key] = lock
        return lock


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


def _fetch_gemini_models_live(default_gemini_models):
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
        result = sorted(set(model_names))
        _set_cached("gemini", result)
        return result
    except Exception as e:
        print(f"Error fetching Gemini models: {str(e)}")
        print("Using default model list.")
        return default_gemini_models


def get_gemini_models():
    # Default models if API call fails. Sourced from
    # https://ai.google.dev/gemini-api/docs/deprecations — last reviewed 2026-05-26.
    default_gemini_models = [
        # Gemini 3.x Models (newest)
        "gemini-3.1-pro-preview",
        "gemini-3-pro-image-preview",      # no shutdown date announced
        "gemini-3.1-flash-image-preview",  # no shutdown date announced
        "gemini-3.5-flash",                # GA, no shutdown date announced
        # Gemini 2.5 Models
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image",
        # Gemini 2.0 Models
        "gemini-2.0-flash",                # shutdown 2026-06-01
        "gemini-2.0-flash-lite",           # shutdown 2026-06-01
        # Image Generation Models
        "imagen-4.0-generate-001",         # shutdown 2026-06-24
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-fast-generate-001",
    ]

    cached = _get_cached("gemini")
    if cached is not None:
        return cached

    with _fetch_lock_for("gemini"):
        # Double-check inside the lock: another caller may have populated
        # the cache while we were waiting.
        cached = _get_cached("gemini")
        if cached is not None:
            return cached
        return _fetch_gemini_models_live(default_gemini_models)


def _fetch_openai_models_live(default_openai_models):
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
        result = sorted(set(model_ids))
        _set_cached("openai", result)
        return result
    except Exception as e:
        print(f"Error fetching OpenAI models: {str(e)}")
        print("Using default OpenAI model list.")
        return default_openai_models


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

    cached = _get_cached("openai")
    if cached is not None:
        return cached

    with _fetch_lock_for("openai"):
        cached = _get_cached("openai")
        if cached is not None:
            return cached
        return _fetch_openai_models_live(default_openai_models)


def _fetch_gemini_image_models_live(fallback_models):
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

        result = sorted(set(image_models))
        _set_cached("gemini_image", result)
        return result
    except Exception as e:
        print(f"Error fetching Gemini image models: {str(e)}")
        print("Using default image model list.")
        return fallback_models


def get_gemini_image_models():
    """
    Dynamically fetches a list of Gemini models that support image generation.

    Models (verified against the deprecation schedule on 2026-05-26):
    - gemini-2.5-flash-image (Nano Banana, GA): shutdown 2026-10-02
    - gemini-3-pro-image-preview (Nano Banana Pro, Preview): no shutdown date
    - gemini-3.1-flash-image-preview (Nano Banana 2, Preview): no shutdown date
    - imagen-4.0-generate-001: shutdown 2026-06-24
    """
    # gemini-2.5-flash-image-preview was shut down 2026-01-15; the GA name is
    # now canonical.
    fallback_models = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "imagen-4.0-generate-001",
    ]

    cached = _get_cached("gemini_image")
    if cached is not None:
        return cached

    with _fetch_lock_for("gemini_image"):
        cached = _get_cached("gemini_image")
        if cached is not None:
            return cached
        return _fetch_gemini_image_models_live(fallback_models)


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
