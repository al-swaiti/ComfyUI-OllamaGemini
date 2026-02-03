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
                    areas = [(box.xyxy[0][2].cpu() - box.xyxy[0][0].cpu()) * (box.xyxy[0][3].cpu() - box.xyxy[0][1].cpu()) 
                             for box in boxes]
                    largest_idx = np.argmax([float(a) for a in areas])
                    boxes = [boxes[largest_idx]]
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    
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
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())
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
    YOLOv8 Segmentation Node
    
    Instance segmentation using standard YOLOv8-seg models.
    Note: Unlike Object Detect, this uses fixed COCO classes (80 categories).
    For text-based segmentation, use CLIPSeg or BEN2+ObjectDetect combo.
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
                "class_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Filter: person, car, dog (leave empty for all)"
                }),
                "model": (["yolov8x-seg", "yolov8l-seg", "yolov8m-seg", "yolov8s-seg", "yolov8n-seg"],),
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

    # COCO class names for filtering
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def segment_objects(self, image, class_filter, model, confidence, return_largest_only=True):
        """Segment objects using standard YOLOv8-seg with optional class filtering"""
        self.load_model(model)
        
        # Parse class filter - match against COCO classes
        filter_classes = []
        if class_filter.strip():
            filter_names = [c.strip().lower() for c in class_filter.split(",") if c.strip()]
            for i, coco_class in enumerate(self.COCO_CLASSES):
                if coco_class.lower() in filter_names:
                    filter_classes.append(i)
        
        processed_extracted = []
        processed_masks = []
        
        for img_tensor in image:
            pil_image = tensor2pil(img_tensor)
            w, h = pil_image.size
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Run prediction with optional class filter
            predict_kwargs = {"conf": confidence, "verbose": False}
            if filter_classes:
                predict_kwargs["classes"] = filter_classes
            
            results = self.model.predict(pil_image, **predict_kwargs)
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
