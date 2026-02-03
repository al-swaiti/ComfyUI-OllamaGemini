"""
YOLOE Object Detection Node for ComfyUI
Text-guided object detection and extraction using YOLOE-26
"""
import torch
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class GeminiObjectDetect:
    """
    YOLOE Object Detection Node
    
    Detect and extract objects from images using text prompts.
    Uses YOLOE for open-vocabulary detection.
    
    Features:
    - Text-based object detection ("cat", "red car", "person")
    - Multiple output modes: mask, bbox, extracted object
    - Batch processing support
    """
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.ultralytics_available = None
    
    def _check_ultralytics(self):
        if self.ultralytics_available is None:
            try:
                from ultralytics import YOLO
                self.ultralytics_available = True
            except ImportError:
                self.ultralytics_available = False
                print("[ObjectDetect] ultralytics not installed. Run: pip install ultralytics")
        return self.ultralytics_available

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "placeholder": "Objects to detect (comma separated)"
                }),
                "model": (["yolov8x-worldv2", "yolov8l-worldv2", "yolov8m-worldv2", "yolov8s-worldv2"],),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "output_mode": (["extracted", "mask", "bbox_overlay", "all"],),
            },
            "optional": {
                "return_largest_only": ("BOOLEAN", {"default": True}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("extracted", "mask", "bbox_image")
    FUNCTION = "detect_objects"
    CATEGORY = "AI API"

    def load_model(self, model_name):
        """Load YOLO-World model"""
        if self.model is None or self.current_model_name != model_name:
            if not self._check_ultralytics():
                raise RuntimeError(
                    "ultralytics not installed. Please run:\n"
                    "pip install ultralytics"
                )
            
            print(f"[ObjectDetect] Loading {model_name}...")
            from ultralytics import YOLO
            
            self.model = YOLO(f"{model_name}.pt")
            self.current_model_name = model_name
            print(f"[ObjectDetect] Model loaded!")

    def detect_objects(self, image, text_prompt, model, confidence, output_mode,
                       return_largest_only=True, padding=0):
        """Detect and extract objects based on text prompt"""
        self.load_model(model)
        
        # Parse text prompt into classes
        classes = [c.strip() for c in text_prompt.split(",") if c.strip()]
        if not classes:
            classes = ["object"]
        
        # Set custom classes for open-vocabulary detection
        self.model.set_classes(classes)
        
        processed_extracted = []
        processed_masks = []
        processed_bbox = []
        
        total_images = image.shape[0]
        is_batch = total_images > 1
        
        if is_batch:
            print(f"[ObjectDetect] Processing {total_images} images...")

        for i, img_tensor in enumerate(image):
            if is_batch and ((i + 1) % 10 == 0 or i == 0):
                print(f"[ObjectDetect] Image {i + 1}/{total_images}")
            
            # Convert to PIL
            pil_image = tensor2pil(img_tensor)
            w, h = pil_image.size
            
            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Run detection
            results = self.model.predict(
                pil_image, 
                conf=confidence,
                verbose=False
            )
            
            result = results[0]
            
            # Create outputs
            extracted_image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            mask_image = Image.new("L", (w, h), 0)
            bbox_image = pil_image.copy()
            
            if len(result.boxes) > 0:
                boxes = result.boxes
                
                # Get largest box if requested
                if return_largest_only and len(boxes) > 1:
                    areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) 
                             for box in boxes]
                    largest_idx = np.argmax(areas)
                    boxes = [boxes[largest_idx]]
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Apply padding
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    # Create mask for this detection
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(mask_image)
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                    
                    # Draw bbox on overlay
                    draw_bbox = ImageDraw.Draw(bbox_image)
                    draw_bbox.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # Get class name and confidence
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = classes[cls_id] if cls_id < len(classes) else "object"
                    label = f"{cls_name}: {conf:.2f}"
                    draw_bbox.text((x1, y1 - 15), label, fill="red")
                
                # Extract object using mask
                pil_rgba = pil_image.convert("RGBA")
                extracted_image.paste(pil_rgba, mask=mask_image)
            
            # Convert to tensors
            extracted_tensor = pil2tensor(extracted_image)
            mask_tensor = pil2tensor(mask_image)
            bbox_tensor = pil2tensor(bbox_image)
            
            processed_extracted.append(extracted_tensor)
            processed_masks.append(mask_tensor)
            processed_bbox.append(bbox_tensor)

        if is_batch:
            print(f"[ObjectDetect] Completed processing {total_images} images!")
        
        return (
            torch.cat(processed_extracted, dim=0),
            torch.cat(processed_masks, dim=0),
            torch.cat(processed_bbox, dim=0)
        )


class GeminiObjectDetectSegment:
    """
    YOLOE Object Detection with Segmentation
    
    Similar to GeminiObjectDetect but outputs precise segmentation masks
    instead of bounding box masks. Requires YOLOE-seg models.
    """
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.ultralytics_available = None
    
    def _check_ultralytics(self):
        if self.ultralytics_available is None:
            try:
                from ultralytics import YOLO
                self.ultralytics_available = True
            except ImportError:
                self.ultralytics_available = False
        return self.ultralytics_available

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "placeholder": "Objects to segment (comma separated)"
                }),
                "model": (["yolov8x-worldv2-seg", "yolov8l-worldv2-seg", "yolov8m-worldv2-seg"],),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "return_largest_only": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("extracted", "mask")
    FUNCTION = "segment_objects"
    CATEGORY = "AI API"

    def load_model(self, model_name):
        if self.model is None or self.current_model_name != model_name:
            if not self._check_ultralytics():
                raise RuntimeError("ultralytics not installed")
            
            print(f"[ObjectDetectSeg] Loading {model_name}...")
            from ultralytics import YOLO
            
            self.model = YOLO(f"{model_name}.pt")
            self.current_model_name = model_name
            print(f"[ObjectDetectSeg] Model loaded!")

    def segment_objects(self, image, text_prompt, model, confidence, return_largest_only=True):
        """Segment objects based on text prompt"""
        self.load_model(model)
        
        classes = [c.strip() for c in text_prompt.split(",") if c.strip()]
        if not classes:
            classes = ["object"]
        
        self.model.set_classes(classes)
        
        processed_extracted = []
        processed_masks = []
        
        for img_tensor in image:
            pil_image = tensor2pil(img_tensor)
            w, h = pil_image.size
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            results = self.model.predict(pil_image, conf=confidence, verbose=False)
            result = results[0]
            
            extracted_image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            mask_image = Image.new("L", (w, h), 0)
            
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.data.cpu().numpy()
                
                if return_largest_only and len(masks) > 1:
                    areas = [m.sum() for m in masks]
                    largest_idx = np.argmax(areas)
                    masks = [masks[largest_idx]]
                
                # Combine all masks
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for m in masks:
                    # Resize mask to image size
                    m_resized = Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))
                    combined_mask = np.maximum(combined_mask, np.array(m_resized))
                
                mask_image = Image.fromarray(combined_mask)
                
                # Extract using mask
                pil_rgba = pil_image.convert("RGBA")
                extracted_image.paste(pil_rgba, mask=mask_image)
            
            processed_extracted.append(pil2tensor(extracted_image))
            processed_masks.append(pil2tensor(mask_image))

        return (
            torch.cat(processed_extracted, dim=0),
            torch.cat(processed_masks, dim=0)
        )


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiObjectDetect": GeminiObjectDetect,
    "GeminiObjectDetectSegment": GeminiObjectDetectSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiObjectDetect": "Object Detect (Text Prompt)",
    "GeminiObjectDetectSegment": "Object Segment (Text Prompt)",
}
