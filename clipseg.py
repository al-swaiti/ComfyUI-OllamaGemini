from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np
import os
import folder_paths
import cv2
from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch") 
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")

# Global variables to store model and processor
_model = None
_processor = None

class GeminiCLIPSeg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "hard_mask")

    FUNCTION = "segment_image"
    def segment_image(self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int) -> torch.Tensor:
        """Create segmentation mask from image and text prompt using CLIPSeg"""
        global _model, _processor

        if _model is None:
            model_name = "clipseg-rd64-refined"
            model_path = os.path.join(folder_paths.models_dir, "clip", model_name)
            
            endpoint = "https://huggingface.co"
            try:
                import requests
                _ = requests.get(endpoint, timeout=5)
            except:
                print("Trying mirror source")
                endpoint = "https://hf-mirror.com"

            if not os.path.exists(model_path):
                print(f"Downloading model {model_name} to {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="CIDAS/clipseg-rd64-refined",
                                local_dir=model_path,
                                local_dir_use_symlinks=False,
                                endpoint=endpoint)
            _processor = CLIPSegProcessor.from_pretrained(model_path)
            _model = CLIPSegForImageSegmentation.from_pretrained(model_path)

        # Convert tensor to numpy array and scale values to 0-255
        image_np = image.numpy().squeeze()
        image_np = (image_np * 255).astype(np.uint8)
        original_size = image_np.shape[:2]
        i = Image.fromarray(image_np, mode="RGB")
        
        input_prc = _processor(text=text, images=[i], return_tensors="pt")
        
        with torch.no_grad():
            outputs = _model(**input_prc)
        preds = outputs.logits.unsqueeze(1)
        tensor = torch.sigmoid(preds[0][0])
        
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        )[0][0]
        
        tensor_thresholded = torch.where(tensor > threshold, tensor, torch.tensor(0, dtype=torch.float))

        tensor_smoothed = gaussian_filter(tensor_thresholded.numpy(), sigma=blur)
        tensor_smoothed = torch.from_numpy(tensor_smoothed)

        mask_normalized = (tensor_smoothed - tensor_smoothed.min()) / (tensor_smoothed.max() - tensor_smoothed.min())

        # Dilate mask using square kernel with given dilation factor
        kernel_size = int(dilation_factor * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dilated = cv2.dilate(mask_normalized.numpy(), kernel, iterations=1)
        mask_dilated = torch.from_numpy(mask_dilated)
        
        # Generate hard edge mask
        hard_mask = torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))
        kernel_size = int(dilation_factor * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        hard_mask = cv2.dilate(hard_mask.numpy(), kernel, iterations=1)
        hard_mask = torch.from_numpy(hard_mask)

        return (mask_dilated, hard_mask)

class GeminiCombineMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "mask1": ("MASK", ), 
                        "mask2": ("MASK", ),
                    },
                }
        
    CATEGORY = "image"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "combine_masks"
            
    def combine_masks(self,  mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        combined_mask = mask1 + mask2
        combined_mask = torch.clamp(combined_mask, 0, 1)
        
        return (combined_mask,)

NODE_CLASS_MAPPINGS = {
    "GeminiCLIPSeg": GeminiCLIPSeg,
    "GeminiCombineSegMasks": GeminiCombineMasks,
}