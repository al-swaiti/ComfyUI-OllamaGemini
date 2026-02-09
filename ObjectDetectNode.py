"""
Ultra Object Detection Node for ComfyUI
========================================
STANDALONE implementation - NO LayerStyle dependencies!

Uses latest 2025/2026 models:
- Detection: YOLO-World V2.1 (ultralytics)
- Segmentation: SAM2.1 (ultralytics)
- Matting: ViTMatte / BiRefNet / Guided Filter

Author: Gemini AI Assistant
"""
import os
import copy
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import math
import warnings


try:
    import folder_paths
except ImportError:
    folder_paths = None

warnings.filterwarnings("ignore", category=FutureWarning)

def search_for_model(model_filename, model_type="ultralytics"):
    """
    Smart search for model file in standard ComfyUI model directories.
    Prioritizes specific folders (e.g., models/sams) but falls back to root/ultralytics.
    """
    if folder_paths is None:
        return model_filename
        
    # getting paths list for model type
    if model_type == "sams":
        # Check standard ComfyUI/models/sams
        sams_paths = folder_paths.get_folder_paths("sams")
        if sams_paths:
            for path in sams_paths:
                full_path = os.path.join(path, model_filename)
                if os.path.exists(full_path):
                    return full_path
    
    # Check ultralytics/yolo specific
    if model_type == "ultralytics":
        ultralytics_paths = folder_paths.get_folder_paths("ultralytics")
        if ultralytics_paths:
            for path in ultralytics_paths:
                # Check root of ultralytics folder
                full_path = os.path.join(path, model_filename)
                if os.path.exists(full_path):
                    return full_path
                # Check bbox/segm/yolo subfolders
                for sub in ["bbox", "segm", "yolo"]:
                    full_path_sub = os.path.join(path, sub, model_filename)
                    if os.path.exists(full_path_sub):
                        return full_path_sub
    
    # Fallback 1: Check ComfyUI/models/checkpoints
    ckpt_paths = folder_paths.get_folder_paths("checkpoints")
    if ckpt_paths:
        for path in ckpt_paths:
            full_path = os.path.join(path, model_filename)
            if os.path.exists(full_path):
                return full_path
                
    # Fallback 2: Check ComfyUI/models root
    # We infer models dir from checkpoints path if available
    if ckpt_paths:
        models_dir = os.path.dirname(ckpt_paths[0])
        models_root_path = os.path.join(models_dir, model_filename)
        if os.path.exists(models_root_path):
            return models_root_path

    # Fallback 3: Check ComfyUI root
    # Infer ComfyUI root from models dir or current file structure
    try:
        # Assuming we are in custom_nodes/ComfyUI-OllamaGemini/ObjectDetectNode.py
        # root is ../../../
        comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        full_path_root = os.path.join(comfy_root, model_filename)
        if os.path.exists(full_path_root):
            return full_path_root
    except:
        pass
        
    # Not found - return filename to let ultralytics handle download/error
    return model_filename




# ============================================================================
# Configuration
# ============================================================================

# Detection models - Open vocabulary with text prompts
DETECTION_MODELS = {
    # YOLOE-26 - Latest (Jan 2026) - Open vocab on YOLO26 architecture
    "yoloe-26x-seg (Latest Best)": "yoloe-26x-seg.pt",
    "yoloe-26l-seg (Latest Balanced)": "yoloe-26l-seg.pt",
    # YOLO-World V2.1 - Proven stable
    "yolov8x-worldv2 (Stable Best)": "yolov8x-worldv2.pt",
    "yolov8l-worldv2 (Stable Balanced)": "yolov8l-worldv2.pt",
    "yolov8m-worldv2 (Fast)": "yolov8m-worldv2.pt",
    "yolov8s-worldv2 (Fastest)": "yolov8s-worldv2.pt",
}

# Segmentation models (SAM via ultralytics)
SAM_MODELS = {
    # SAM3 - Latest (Nov 2025) - Requires manual download from HuggingFace
    "sam3 (Latest - needs HF download)": "sam3.pt",
    # SAM2.1 - Stable, auto-downloads
    "sam2.1-large (Stable Best)": "sam2.1_l.pt",
    "sam2.1-base+ (Stable Balanced)": "sam2.1_b.pt",
    "sam2.1-small (Fast)": "sam2.1_s.pt",
    "sam2.1-tiny (Fastest)": "sam2.1_t.pt",
}

# Matting models (ViTMatte via transformers)
VITMATTE_MODELS = {
    "vitmatte-small (Fast)": "hustvl/vitmatte-small-composition-1k",
    "vitmatte-base (Quality)": "hustvl/vitmatte-base-composition-1k",
}

# BiRefNet models - Best for hair/edge matting
BIREFNET_MODELS = {
    # BiRefNet-dynamic - Latest (Mar 2025) - Dynamic 256-2304 resolution
    "BiRefNet-dynamic (Latest Best)": "ZhengPeng7/BiRefNet_dynamic",
    "BiRefNet-matting (Best Hair/Edges)": "ZhengPeng7/BiRefNet-matting",
    "BiRefNet-HR-matting (High-Res 2K)": "ZhengPeng7/BiRefNet_HR-matting",
    "RMBG-2.0 (General)": "briaai/RMBG-2.0",
}

# Matting methods
MATTING_METHODS = [
    "BiRefNet-matting (Best Quality)",  # DEFAULT - best for hair/edges
    "ViTMatte (Trimap-based)",
    "Guided Filter (Fast)",
    "None (Raw Mask)",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Utilities
# ============================================================================

def tensor2pil(tensor):
    """Convert tensor to PIL Image, handling various shapes."""
    if isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        # Handle 4D tensor (B, C, H, W) or (B, H, W, C)
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        # Handle 3D tensor (C, H, W) where C is 1, 3, or 4
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            # Squeeze single channel
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
        arr = tensor.cpu().numpy()
    
    # Normalize to 0-255
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil2tensor(image):
    """Convert PIL Image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image):
    """Convert image to mask tensor."""
    if isinstance(image, Image.Image):
        image = pil2tensor(image.convert('L'))
    return image.squeeze(0) if image.ndim == 4 else image

def mask2image(mask):
    """Convert mask to RGB image."""
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).repeat(1, 1, 3)
        return tensor2pil(mask)
    return mask

def RGB2RGBA(image, mask):
    """Combine RGB image with alpha mask."""
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def log(message, message_type='info'):
    """Log message with prefix."""
    prefix = "[UltraDetect]"
    if message_type == 'error':
        print(f"{prefix} ERROR: {message}")
    elif message_type == 'warning':
        print(f"{prefix} WARNING: {message}")
    else:
        print(f"{prefix} {message}")

def clear_memory():
    """Clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ============================================================================
# Model Manager (Handles all model loading/caching)
# ============================================================================

class ModelManager:
    """Singleton model manager for efficient model caching."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_cache()
        return cls._instance
    
    def _init_cache(self):
        self.yolo_model = None
        self.yolo_name = None
        self.sam_model = None
        self.sam_name = None
        self.sam3_semantic = None  # SAM3SemanticPredictor for text prompts
        self.sam3_semantic_name = None
        self.vitmatte_model = None
        self.vitmatte_name = None
        self.birefnet_model = None
        self.birefnet_name = None
    
    def get_yolo(self, model_key):
        """Load YOLO detection model (YOLOE or YOLO-World)."""
        if self.yolo_name != model_key:
            model_file = DETECTION_MODELS[model_key]
            
            # Smart path search
            model_path = search_for_model(model_file, "ultralytics")
            
            # Use YOLOE class for yoloe models, YOLO for yolo-world
            if "yoloe" in model_file.lower():
                from ultralytics import YOLOE
                log(f"Loading YOLOE: {model_key} from {model_path}...")
                self.yolo_model = YOLOE(model_path)
            else:
                from ultralytics import YOLO
                log(f"Loading YOLO-World: {model_key} from {model_path}...")
                self.yolo_model = YOLO(model_path)
            
            self.yolo_name = model_key
            log(f"Detection model loaded!")
        return self.yolo_model
    
    def get_sam(self, model_key):
        """Load SAM model."""
        if self.sam_name != model_key:
            from ultralytics import SAM
            model_file = SAM_MODELS[model_key]
            
            # Smart path search
            model_path = search_for_model(model_file, "sams")
            
            log(f"Loading SAM: {model_key} from {model_path}...")
            self.sam_model = SAM(model_path)
            self.sam_name = model_key
            log(f"SAM loaded!")
        return self.sam_model
    
    def get_sam3_semantic(self, model_key="sam3 (Latest - needs HF download)"):
        """Load SAM3SemanticPredictor for direct text-based segmentation."""
        if self.sam3_semantic_name != model_key:
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor
            except ImportError:
                log("SAM3SemanticPredictor not available. Update ultralytics: pip install -U ultralytics", 'error')
                return None
            
            model_file = SAM_MODELS.get(model_key, "sam3.pt")
            model_path = search_for_model(model_file, "sams")
            
            log(f"Loading SAM3 Semantic Predictor from {model_path}...")
            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=model_path,
                half=True,
                verbose=False,
            )
            self.sam3_semantic = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_semantic_name = model_key
            log(f"SAM3 Semantic Predictor loaded!")
        return self.sam3_semantic
    
    def get_vitmatte(self, model_key):
        """Load ViTMatte model."""
        if self.vitmatte_name != model_key:
            from transformers import VitMatteImageProcessor, VitMatteForImageMatting
            model_id = VITMATTE_MODELS[model_key]
            log(f"Loading ViTMatte: {model_key}...")
            processor = VitMatteImageProcessor.from_pretrained(model_id)
            model = VitMatteForImageMatting.from_pretrained(model_id).to(device)
            self.vitmatte_model = {"processor": processor, "model": model}
            self.vitmatte_name = model_key
            log(f"ViTMatte loaded!")
        return self.vitmatte_model
    
    def get_birefnet(self, model_key="BiRefNet-matting (Best Hair/Edges)"):
        """Load BiRefNet model - supports multiple variants."""
        if self.birefnet_name != model_key or self.birefnet_model is None:
            from transformers import AutoModelForImageSegmentation
            model_id = BIREFNET_MODELS.get(model_key, "ZhengPeng7/BiRefNet-matting")
            log(f"Loading BiRefNet: {model_key}...")
            self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                model_id, trust_remote_code=True
            ).to(device)
            self.birefnet_model.eval()
            self.birefnet_name = model_key
            log(f"BiRefNet loaded!")
        return self.birefnet_model
    
    def clear(self):
        """Clear all cached models."""
        self.yolo_model = None
        self.yolo_name = None
        self.sam_model = None
        self.sam_name = None
        self.sam3_semantic = None
        self.sam3_semantic_name = None
        self.vitmatte_model = None
        self.vitmatte_name = None
        self.birefnet_model = None
        self.birefnet_name = None
        clear_memory()

# Global model manager
models = ModelManager()

# ============================================================================
# Detection Functions
# ============================================================================

def detect_yolo_world(image_pil, prompt, threshold=0.25):
    """Detect objects using YOLO-World with text prompt."""
    model = models.get_yolo(list(DETECTION_MODELS.keys())[0])
    
    # Set classes from prompt
    classes = [c.strip() for c in prompt.split(",")]
    model.set_classes(classes)
    
    # Run inference
    results = model.predict(image_pil, conf=threshold, verbose=False)
    
    boxes = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
    
    return boxes

# ============================================================================
# Segmentation Functions
# ============================================================================

def segment_sam2(image_pil, boxes):
    """Segment objects using SAM2.1 with bounding boxes."""
    if not boxes:
        return None
    
    model = models.get_sam(list(SAM_MODELS.keys())[0])
    
    # Convert boxes to tensor
    bboxes = torch.tensor(boxes, device=device)
    
    # Run SAM inference
    results = model.predict(image_pil, bboxes=bboxes, verbose=False)
    
    # Combine all masks
    combined_mask = None
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy()
                if combined_mask is None:
                    combined_mask = mask_np
                else:
                    combined_mask = np.maximum(combined_mask, mask_np)
    
    return combined_mask

# ============================================================================
# Matting Functions
# ============================================================================

def generate_trimap(mask_np, erode_size=6, dilate_size=6):
    """Generate trimap from binary mask."""
    # Ensure mask is binary 0-255
    mask_uint8 = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1 else mask_np.astype(np.uint8)
    
    # Create kernels
    erode_kernel = np.ones((erode_size, erode_size), np.uint8)
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    
    # Erode for definite foreground, dilate for definite background
    eroded = cv2.erode(mask_uint8, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=5)
    
    # Build trimap: 0=background, 128=unknown, 255=foreground
    trimap = np.zeros_like(mask_uint8)
    trimap[dilated > 127] = 128
    trimap[eroded > 127] = 255
    
    return Image.fromarray(trimap, mode='L')

def refine_vitmatte(image_pil, mask_np, erode_size=6, dilate_size=6, max_megapixels=2.0):
    """Refine mask using ViTMatte."""
    vitmatte = models.get_vitmatte(list(VITMATTE_MODELS.keys())[0])
    processor = vitmatte["processor"]
    model = vitmatte["model"]
    
    # Generate trimap
    trimap = generate_trimap(mask_np, erode_size, dilate_size)
    
    # Resize if too large
    max_pixels = max_megapixels * 1048576
    w, h = image_pil.size
    if w * h > max_pixels:
        ratio = math.sqrt(max_pixels / (w * h))
        new_w, new_h = int(w * ratio), int(h * ratio)
        image_resized = image_pil.resize((new_w, new_h), Image.BILINEAR)
        trimap_resized = trimap.resize((new_w, new_h), Image.BILINEAR)
    else:
        image_resized = image_pil
        trimap_resized = trimap
        new_w, new_h = w, h
    
    # Prepare inputs
    inputs = processor(images=image_resized, trimaps=trimap_resized, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get alpha
    alpha = outputs.alphas[0, 0].cpu().numpy()
    
    # Resize back if needed
    if w * h > max_pixels:
        alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), mode='L')
        alpha_pil = alpha_pil.resize((w, h), Image.BILINEAR)
        alpha = np.array(alpha_pil).astype(np.float32) / 255.0
    
    clear_memory()
    return alpha

def refine_birefnet(image_pil, model_key="BiRefNet-matting (Best Hair/Edges)"):
    """Refine mask using BiRefNet models - best for hair and fine edges."""
    from torchvision import transforms
    
    model = models.get_birefnet(model_key)
    
    # Prepare image - use higher resolution for BiRefNet-HR-matting
    if "HR" in model_key:
        target_size = (2048, 2048)
    else:
        target_size = (1024, 1024)
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_pil.convert('RGB')).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()
    
    # Process output
    pred = preds[0].squeeze().cpu().numpy()
    
    # Resize to original
    mask_pil = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
    mask_pil = mask_pil.resize(image_pil.size, Image.BILINEAR)
    
    clear_memory()
    return np.array(mask_pil).astype(np.float32) / 255.0

def refine_guided_filter(image_pil, mask_np, radius=10):
    """Refine mask using Guided Filter."""
    from cv2.ximgproc import guidedFilter
    
    # Convert to numpy
    image_np = np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0
    mask_np = mask_np.astype(np.float32)
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0
    
    # Apply guided filter
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
    refined = guidedFilter(image_np, mask_rgb, radius, 0.01)
    
    return refined[:, :, 0]

def histogram_remap(mask_np, black_point=0.15, white_point=0.99):
    """Remap mask histogram for better contrast."""
    bp = min(black_point, white_point - 0.001)
    scale = 1.0 / (white_point - bp)
    return np.clip((mask_np - bp) * scale, 0.0, 1.0)

# ============================================================================
# Main Node Class
# ============================================================================

class GeminiUltraDetect:
    """
    Ultra Object Detection Node
    ---------------------------
    Standalone implementation using latest 2025/2026 models:
    - YOLO-World V2.1 for detection
    - SAM2.1 for segmentation
    - ViTMatte/BiRefNet for matting
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "person", "multiline": False}),
            },
            "optional": {
                "detection_model": (list(DETECTION_MODELS.keys()), {"default": list(DETECTION_MODELS.keys())[0]}),
                "sam_model": (list(SAM_MODELS.keys()), {"default": list(SAM_MODELS.keys())[0]}),
                "matting_method": (MATTING_METHODS, {"default": MATTING_METHODS[0]}),
                "birefnet_model": (list(BIREFNET_MODELS.keys()), {"default": list(BIREFNET_MODELS.keys())[0]}),
                "vitmatte_model": (list(VITMATTE_MODELS.keys()), {"default": list(VITMATTE_MODELS.keys())[0]}),
                "detection_threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 1.0, "step": 0.01}),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "cache_models": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BBOXES")
    RETURN_NAMES = ("image", "black_masked", "mask", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "AI API/Detection"
    
    def detect(self, image, prompt,
               detection_model=None, sam_model=None, matting_method=None, 
               birefnet_model=None, vitmatte_model=None,
               detection_threshold=0.25, detail_erode=6, detail_dilate=6,
               black_point=0.15, white_point=0.99, max_megapixels=2.0, cache_models=True):
        
        # Default values
        if detection_model is None:
            detection_model = list(DETECTION_MODELS.keys())[0]
        if sam_model is None:
            sam_model = list(SAM_MODELS.keys())[0]
        if matting_method is None:
            matting_method = MATTING_METHODS[0]
        if birefnet_model is None:
            birefnet_model = list(BIREFNET_MODELS.keys())[0]
        if vitmatte_model is None:
            vitmatte_model = list(VITMATTE_MODELS.keys())[0]
        
        ret_images = []
        ret_masks = []
        all_bboxes = []
        
        for i in range(image.shape[0]):
            # Get single image
            img_tensor = image[i]
            pil_image = tensor2pil(img_tensor).convert('RGB')
            
            log(f"Detecting '{prompt}'...")
            
            # Check if using SAM3 with direct text prompts (bypasses YOLO)
            use_sam3_direct = "sam3" in sam_model.lower()
            
            if use_sam3_direct:
                # SAM3 Direct Text Segmentation - can detect concepts like "sun", "lake", etc.
                try:
                    sam3_predictor = models.get_sam3_semantic(sam_model)
                    if sam3_predictor is not None:
                        log(f"Using SAM3 direct text segmentation...")
                        classes = [c.strip() for c in prompt.split(",")]
                        
                        # Set image
                        sam3_predictor.set_image(pil_image)
                        
                        # Segment with text prompts
                        results = sam3_predictor(text=classes)
                        
                        combined_mask = None
                        boxes = []
                        for result in results:
                            if result.masks is not None:
                                for mask in result.masks.data:
                                    mask_np = mask.cpu().numpy()
                                    if combined_mask is None:
                                        combined_mask = mask_np
                                    else:
                                        combined_mask = np.maximum(combined_mask, mask_np)
                            if result.boxes is not None:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        
                        all_bboxes.append(boxes)
                        
                        if combined_mask is None:
                            log(f"No objects found with SAM3.")
                            h, w = pil_image.size[1], pil_image.size[0]
                            empty_mask = torch.zeros((h, w), dtype=torch.float32)
                            ret_masks.append(empty_mask)
                            ret_images.append(pil2tensor(pil_image))
                            continue
                        
                        log(f"SAM3 found {len(boxes)} object(s)!")
                        
                        # Resize mask to image size if needed
                        h, w = pil_image.size[1], pil_image.size[0]
                        if combined_mask.shape != (h, w):
                            combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Apply matting refinement
                        mask_tensor = torch.from_numpy(combined_mask).float()
                        
                        # Skip to matting step
                        if matting_method == "BiRefNet-matting (Best Quality)":
                            log(f"Refining with {matting_method}...")
                            # BiRefNet does full-image segmentation
                            birefnet_mask = refine_birefnet(pil_image, birefnet_model)
                            # Combine SAM3 detection with BiRefNet edge refinement
                            # Use SAM3 as the region, BiRefNet for edge quality
                            combined = combined_mask * birefnet_mask  # Intersection
                            # If intersection is too small, use SAM3 mask weighted
                            if combined.sum() < combined_mask.sum() * 0.3:
                                # BiRefNet didn't find the same region, use SAM3 with edge refinement
                                refined = refine_guided_filter(pil_image, combined_mask)
                                mask_tensor = torch.from_numpy(refined).float()
                            else:
                                mask_tensor = torch.from_numpy(combined).float()
                        elif matting_method == "Guided Filter (Fast)":
                            refined = refine_guided_filter(pil_image, combined_mask)
                            mask_tensor = torch.from_numpy(refined).float()
                        
                        ret_masks.append(mask_tensor)
                        
                        # Create RGBA output
                        mask_pil = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))
                        rgba = RGB2RGBA(pil_image, mask_pil)
                        ret_images.append(pil2tensor(rgba))
                        
                        log(f"Done!")
                        continue
                    else:
                        log("SAM3 predictor not available, falling back to YOLO+SAM...", 'warning')
                except Exception as e:
                    log(f"SAM3 direct segmentation failed: {e}, falling back to YOLO+SAM...", 'warning')
            
            # Step 1: Detection with YOLO-World (fallback or primary)
            try:
                yolo = models.get_yolo(detection_model)
                classes = [c.strip() for c in prompt.split(",")]
                
                # Move model to device BEFORE set_classes
                target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                yolo.to(target_device)
                
                # Set classes - YOLOE requires get_text_pe for text embeddings!
                model_file = DETECTION_MODELS.get(detection_model, "")
                if "yoloe" in model_file.lower() and hasattr(yolo, 'get_text_pe'):
                    # YOLOE needs text prompt embeddings
                    text_pe = yolo.get_text_pe(classes)
                    yolo.set_classes(classes, text_pe)
                    log(f"YOLOE: Set {len(classes)} classes with text embeddings")
                else:
                    # YOLO-World uses simpler set_classes
                    yolo.set_classes(classes)
                
                # CRITICAL: Move text features to same device as model
                # This fixes the "tensors on different devices" error
                if hasattr(yolo.model, 'txt_feats') and yolo.model.txt_feats is not None:
                    yolo.model.txt_feats = yolo.model.txt_feats.to(target_device)
                if hasattr(yolo.model, 'model') and hasattr(yolo.model.model, 'txt_feats'):
                    if yolo.model.model.txt_feats is not None:
                        yolo.model.model.txt_feats = yolo.model.model.txt_feats.to(target_device)
                
                # Run prediction with explicit device
                results = yolo.predict(pil_image, conf=detection_threshold, verbose=False, device=target_device)
                
                boxes = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            except Exception as e:
                log(f"Detection failed: {e}", 'error')
                boxes = []
            
            all_bboxes.append(boxes)
            
            if not boxes:
                log(f"No objects found.")
                log(f"TIP: YOLO-World can't detect abstract concepts like 'sun', 'lake'. Try SAM3 for text-based segmentation!", 'warning')
                # Return empty mask
                h, w = pil_image.size[1], pil_image.size[0]
                empty_mask = torch.zeros((h, w), dtype=torch.float32)
                ret_masks.append(empty_mask)
                ret_images.append(pil2tensor(pil_image))
                continue
            
            log(f"Found {len(boxes)} object(s). Segmenting...")
            
            # Step 2: Segmentation with SAM2.1
            try:
                sam = models.get_sam(sam_model)
                bboxes_tensor = torch.tensor(boxes, device=device)
                results = sam.predict(pil_image, bboxes=bboxes_tensor, verbose=False)
                
                combined_mask = None
                for result in results:
                    if result.masks is not None:
                        for mask in result.masks.data:
                            mask_np = mask.cpu().numpy()
                            if combined_mask is None:
                                combined_mask = mask_np
                            else:
                                combined_mask = np.maximum(combined_mask, mask_np)
            except Exception as e:
                log(f"Segmentation failed: {e}", 'error')
                combined_mask = None
            
            if combined_mask is None:
                log(f"Segmentation failed.")
                h, w = pil_image.size[1], pil_image.size[0]
                empty_mask = torch.zeros((h, w), dtype=torch.float32)
                ret_masks.append(empty_mask)
                ret_images.append(pil2tensor(pil_image))
                continue
            
            # Step 3: Matting/Refinement
            log(f"Refining with {matting_method}...")
            
            try:
                if "ViTMatte" in matting_method:
                    # Load specific vitmatte model
                    models.get_vitmatte(vitmatte_model)
                    refined_mask = refine_vitmatte(
                        pil_image, combined_mask,
                        erode_size=detail_erode, dilate_size=detail_dilate,
                        max_megapixels=max_megapixels
                    )
                    refined_mask = histogram_remap(refined_mask, black_point, white_point)
                    
                elif "BiRefNet" in matting_method:
                    refined_mask = refine_birefnet(pil_image, birefnet_model)
                    # Combine with SAM mask for better edge preservation
                    refined_mask = np.maximum(refined_mask, combined_mask * 0.3)
                    refined_mask = histogram_remap(refined_mask, black_point, white_point)
                    
                elif "Guided Filter" in matting_method:
                    refined_mask = refine_guided_filter(pil_image, combined_mask, radius=detail_erode)
                    refined_mask = histogram_remap(refined_mask, black_point, white_point)
                    
                else:  # None - raw mask
                    refined_mask = combined_mask
                    if refined_mask.max() <= 1:
                        pass  # Already normalized
                    else:
                        refined_mask = refined_mask / 255.0
            except Exception as e:
                log(f"Matting failed: {e}, using raw mask", 'warning')
                refined_mask = combined_mask
                if refined_mask.max() > 1:
                    refined_mask = refined_mask / 255.0
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(refined_mask.astype(np.float32))
            
            # Create RGBA image - ensure mask matches image size
            mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8), mode='L')
            
            # Resize mask to match image if sizes differ
            if mask_pil.size != pil_image.size:
                mask_pil = mask_pil.resize(pil_image.size, Image.BILINEAR)
                # Also update tensor
                mask_tensor = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
            
            rgba_image = RGB2RGBA(pil_image, mask_pil)
            
            ret_masks.append(mask_tensor)
            ret_images.append(pil2tensor(rgba_image))
            
            log(f"Done!")
        
        # Clear models if not caching
        if not cache_models:
            models.clear()
        
        # Stack results
        if ret_images:
            output_images = torch.cat(ret_images, dim=0)
            output_masks = torch.stack(ret_masks, dim=0)
            
            # Create black-masked images (original with detected area blacked out)
            black_masked_list = []
            for i in range(image.shape[0]):
                orig_img = image[i]  # [H, W, C]
                if i < len(ret_masks):
                    mask = ret_masks[i]  # [H, W]
                    # Resize mask if needed
                    if mask.shape[0] != orig_img.shape[0] or mask.shape[1] != orig_img.shape[1]:
                        mask = torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=(orig_img.shape[0], orig_img.shape[1]),
                            mode='bilinear'
                        ).squeeze()
                    # Invert mask (1 where we want to keep, 0 where detected)
                    inv_mask = 1.0 - mask.unsqueeze(-1)  # [H, W, 1]
                    # Apply mask (black out detected area)
                    black_masked = orig_img * inv_mask
                    black_masked_list.append(black_masked.unsqueeze(0))
                else:
                    black_masked_list.append(orig_img.unsqueeze(0))
            output_black_masked = torch.cat(black_masked_list, dim=0)
        else:
            # Fallback
            output_images = image
            output_black_masked = image
            h, w = image.shape[1], image.shape[2]
            output_masks = torch.zeros((1, h, w), dtype=torch.float32)
        
        return (output_images, output_black_masked, output_masks, all_bboxes)


# ============================================================================
# Helper Nodes
# ============================================================================

class GeminiDrawBBoxMask:
    """Draw bounding boxes as mask."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOXES",),
            },
            "optional": {
                "grow": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "draw"
    CATEGORY = "AI API/Detection"
    
    def draw(self, image, bboxes, grow=0.0):
        ret = []
        for i, img in enumerate(image):
            pil_img = tensor2pil(img)
            w, h = pil_img.size
            mask = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(mask)
            
            bb = bboxes[i] if i < len(bboxes) else []
            for b in bb:
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if grow != 0:
                    x1 -= bw * grow
                    y1 -= bh * grow
                    x2 += bw * grow
                    y2 += bh * grow
                d.rectangle([max(0, x1), max(0, y1), min(w, x2), min(h, y2)], fill=255)
            
            ret.append(pil2tensor(mask))
        
        return (torch.cat(ret, dim=0),)


class GeminiBBoxJoin:
    """Join multiple bounding box lists."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes_1": ("BBOXES",),
            },
            "optional": {
                "bboxes_2": ("BBOXES",),
                "bboxes_3": ("BBOXES",),
            }
        }
    
    RETURN_TYPES = ("BBOXES",)
    FUNCTION = "join"
    CATEGORY = "AI API/Detection"
    
    def join(self, bboxes_1, bboxes_2=None, bboxes_3=None):
        out = []
        max_len = max(
            len(bboxes_1),
            len(bboxes_2) if bboxes_2 else 0,
            len(bboxes_3) if bboxes_3 else 0
        )
        
        for i in range(max_len):
            combined = []
            if i < len(bboxes_1):
                combined.extend(bboxes_1[i])
            if bboxes_2 and i < len(bboxes_2):
                combined.extend(bboxes_2[i])
            if bboxes_3 and i < len(bboxes_3):
                combined.extend(bboxes_3[i])
            out.append(combined)
        
        return (out,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "GeminiUltraDetect": GeminiUltraDetect,
    "GeminiDrawBBoxMask": GeminiDrawBBoxMask,
    "GeminiBBoxJoin": GeminiBBoxJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiUltraDetect": "Ultra Detect (Latest AI Models)",
    "GeminiDrawBBoxMask": "Draw BBox Mask",
    "GeminiBBoxJoin": "BBox Join",
}
