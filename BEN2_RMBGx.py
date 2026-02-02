"""
BEN2 (Background Erase Network) Node for ComfyUI
State-of-the-art background removal with superior hair/fur segmentation
"""
import torch
from PIL import Image
import numpy as np
import warnings

# Suppress warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('high')


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class GeminiBEN2_RMBG:
    """
    BEN2 Background Removal Node
    
    Uses the BEN2 (Background Erase Network) model for high-quality
    foreground segmentation. Excels at:
    - Hair and fur segmentation
    - 4K image processing
    - Edge refinement
    """
    
    def __init__(self):
        self.model = None
        self.ben2_available = None
    
    def _check_ben2_available(self):
        """Check if ben2 package is installed"""
        if self.ben2_available is None:
            try:
                from ben2 import AutoModel
                self.ben2_available = True
            except ImportError:
                self.ben2_available = False
                print("[BEN2] Package not installed. Run: pip install git+https://github.com/PramaLLC/BEN2.git")
        return self.ben2_available

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "refine_foreground": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "AI API"

    def load_model(self):
        """Load BEN2 model from HuggingFace"""
        if self.model is None:
            if not self._check_ben2_available():
                raise RuntimeError(
                    "BEN2 package not installed. Please run:\n"
                    "pip install git+https://github.com/PramaLLC/BEN2.git"
                )
            
            print("[BEN2] Loading model from PramaLLC/BEN2...")
            from ben2 import AutoModel
            
            self.model = AutoModel.from_pretrained("PramaLLC/BEN2")
            self.model.to(device).eval()
            print("[BEN2] Model loaded successfully!")

    def process_image(self, image, refine_foreground=False):
        """Process images through BEN2"""
        self.load_model()
        
        processed_images = []
        processed_masks = []

        for img_tensor in image:
            # Convert tensor to PIL
            orig_image = tensor2pil(img_tensor)
            w, h = orig_image.size
            
            # Ensure RGB
            if orig_image.mode != 'RGB':
                orig_image = orig_image.convert('RGB')
            
            # Run BEN2 inference
            with torch.no_grad():
                foreground = self.model.inference(
                    orig_image, 
                    refine_foreground=refine_foreground
                )
            
            # foreground is RGBA PIL image
            # Extract mask from alpha channel
            if foreground.mode == 'RGBA':
                # Get alpha channel as mask
                r, g, b, a = foreground.split()
                mask = a
                # Create RGBA output
                new_im = foreground
            else:
                # Fallback if not RGBA
                mask = Image.new('L', (w, h), 255)
                new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
                new_im.paste(orig_image, mask=mask)
            
            # Convert to tensors
            new_im_tensor = pil2tensor(new_im)
            mask_tensor = pil2tensor(mask)

            processed_images.append(new_im_tensor)
            processed_masks.append(mask_tensor)

        return torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0)


class GeminiBEN2_Video:
    """
    BEN2 Video Segmentation Node
    
    Process video files with BEN2 for background removal.
    Outputs WebM with alpha or MP4 with custom background color.
    """
    
    def __init__(self):
        self.model = None
        self.ben2_available = None
    
    def _check_ben2_available(self):
        if self.ben2_available is None:
            try:
                from ben2 import AutoModel
                self.ben2_available = True
            except ImportError:
                self.ben2_available = False
        return self.ben2_available

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "./", "multiline": False}),
                "refine_foreground": ("BOOLEAN", {"default": False}),
                "output_webm": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 3}),
            },
            "optional": {
                "bg_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "bg_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "bg_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "process_video"
    CATEGORY = "AI API"

    def load_model(self):
        if self.model is None:
            if not self._check_ben2_available():
                raise RuntimeError(
                    "BEN2 package not installed. Please run:\n"
                    "pip install git+https://github.com/PramaLLC/BEN2.git"
                )
            
            print("[BEN2] Loading model for video processing...")
            import BEN2 as ben2_module
            
            self.model = ben2_module.BEN_Base().to(device).eval()
            # Model weights are auto-downloaded
            print("[BEN2] Video model loaded!")

    def process_video(self, video_path, output_path, refine_foreground, 
                      output_webm, batch_size, bg_r=0, bg_g=255, bg_b=0):
        """Process video with BEN2"""
        self.load_model()
        
        print(f"[BEN2] Processing video: {video_path}")
        
        self.model.segment_video(
            video_path=video_path,
            output_path=output_path,
            fps=0,  # Auto-detect
            refine_foreground=refine_foreground,
            batch=batch_size,
            print_frames_processed=True,
            webm=output_webm,
            rgb_value=(bg_r, bg_g, bg_b)
        )
        
        ext = "webm" if output_webm else "mp4"
        output_file = f"{output_path}/foreground.{ext}"
        print(f"[BEN2] Video saved to: {output_file}")
        
        return (output_file,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiBEN2_RMBG": GeminiBEN2_RMBG,
    "GeminiBEN2_Video": GeminiBEN2_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiBEN2_RMBG": "BEN2 Background Removal",
    "GeminiBEN2_Video": "BEN2 Video Segmentation",
}
