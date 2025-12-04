import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize
import numpy as np
from transformers import AutoModelForImageSegmentation
import os
import warnings
import sys

# Suppress timm deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('high')

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image: Image.Image, size=(1024, 1024)) -> Image.Image:
    image = image.convert('RGB')
    return image.resize(size, Image.BILINEAR)

class GeminiBRIA_RMBG:
    def __init__(self):
        self.model = None
        self.current_version = None
        self.transformers_version = None
        self._check_transformers_version()

    def _check_transformers_version(self):
        """Check transformers version for compatibility"""
        try:
            import transformers
            version_parts = transformers.__version__.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            self.transformers_version = (major, minor)
        except:
            self.transformers_version = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["1.4", "2.0", "auto"],),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "AI API"

    def load_model(self, model_version):
        # Auto-detect best version based on transformers compatibility
        if model_version == "auto":
            if self.transformers_version and self.transformers_version[0] >= 4 and self.transformers_version[1] >= 49:
                model_version = "2.0"
            else:
                model_version = "1.4"
                print(f"Auto-detected: Using RMBG-1.4 due to transformers version compatibility")

        # Only load if we need to (version changed or model not loaded)
        if self.model is None or self.current_version != model_version:
            try:
                if model_version == "1.4":
                    print("Loading RMBG-1.4...")
                    self.model = AutoModelForImageSegmentation.from_pretrained(
                        "briaai/RMBG-1.4", trust_remote_code=True
                    )
                elif model_version == "2.0":
                    print("Loading RMBG-2.0...")
                    try:
                        # Temporarily suppress warnings during model loading
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            self.model = AutoModelForImageSegmentation.from_pretrained(
                                "briaai/RMBG-2.0",
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True
                            )
                    except (AttributeError, ImportError, RuntimeError) as e:
                        error_msg = str(e)
                        # Check if the error is related to compatibility
                        if any(x in error_msg.lower() for x in ['config', 'attribute', 'transformers', 'timm']):
                            # Instead of falling back, print a clear error and stop execution by raising an exception.
                            print(f"--- ERROR ---")
                            print(f"Failed to load RMBG-2.0 due to package incompatibility.")
                            print(f"RMBG-2.0 requires transformers version 4.49.0 or newer.")
                            print(f"Note: As of August 2025, transformers versions >4.55.4 can cause issues.")
                            print(f"Please update your packages by running:")
                            print(f"pip install --upgrade 'transformers>=4.49.0' 'timm>=0.9.0'")
                            print(f"-------------")
                            raise RuntimeError("RMBG-2.0 compatibility error. Check console for instructions.") from e
                        else:
                            # If it's a different error, just re-raise it.
                            raise e
                else:
                    raise ValueError(f"Invalid model version: {model_version}")

                self.model.to(device)
                self.model.eval()
                self.current_version = model_version
                print(f"Successfully loaded RMBG-{self.current_version}")

            except Exception as e:
                # Re-raise any other exception instead of trying a fallback.
                raise e

    def process_image(self, model_version, image):
        # Load model with error handling
        self.load_model(model_version)

        # Use the actual loaded model version
        actual_version = self.current_version

        processed_images = []
        processed_masks = []

        for img_tensor in image:
            orig_image = tensor2pil(img_tensor)
            w, h = orig_image.size

            # Process based on actual model version
            if actual_version == "1.4":
                # RMBG-1.4 processing
                resized_image = resize_image(orig_image)
                im_np = np.array(resized_image)
                im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
                im_tensor = im_tensor.unsqueeze(0)
                im_tensor = im_tensor / 255.0
                im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
                im_tensor = im_tensor.to(device)

                with torch.no_grad():
                    result = self.model(im_tensor)
                    result_tensor = result[0][0]
                    result_tensor = F.interpolate(result_tensor, size=(h, w), mode='bilinear')
                    result_tensor = torch.squeeze(result_tensor, 0)
                    ma = torch.max(result_tensor)
                    mi = torch.min(result_tensor)
                    result_tensor = (result_tensor - mi) / (ma - mi)
                    im_array = (
                        (result_tensor * 255)
                        .permute(1, 2, 0)
                        .cpu()
                        .data.numpy()
                        .astype(np.uint8)
                    )
                    im_array = np.squeeze(im_array)
                    pil_mask_im = Image.fromarray(im_array)

            else:  # 2.0
                # RMBG-2.0 processing
                transform_image = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                im_tensor = transform_image(orig_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    preds = self.model(im_tensor)

                    if isinstance(preds, (list, tuple)):
                        pred_output = preds[-1]
                    else:
                        pred_output = preds

                    pred_sigmoid = pred_output.sigmoid().cpu()
                    pred = pred_sigmoid[0].squeeze()

                    pil_mask_im = transforms.ToPILImage()(pred)
                    pil_mask_im = pil_mask_im.resize((w, h), Image.BILINEAR)

            # Create final images with alpha channel
            new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
            new_im.paste(orig_image, mask=pil_mask_im)

            # Convert back to tensors
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_mask_im)

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        return torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0)


class GeminiBRIA_RMBG_Safe:
    """Safe version that only uses RMBG-1.4 to avoid compatibility issues"""
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "AI API"

    def load_model(self):
        if self.model is None:
            print("Loading RMBG-1.4 (Safe Mode)...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4", trust_remote_code=True
            )
            self.model.to(device)
            self.model.eval()
            print("Successfully loaded RMBG-1.4")

    def process_image(self, image):
        self.load_model()
        processed_images = []
        processed_masks = []

        for img_tensor in image:
            orig_image = tensor2pil(img_tensor)
            w, h = orig_image.size

            # RMBG-1.4 processing
            resized_image = resize_image(orig_image)
            im_np = np.array(resized_image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = im_tensor.unsqueeze(0)
            im_tensor = im_tensor / 255.0
            im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            im_tensor = im_tensor.to(device)

            # Process image
            with torch.no_grad():
                result = self.model(im_tensor)
                result_tensor = result[0][0]
                result_tensor = F.interpolate(result_tensor, size=(h, w), mode='bilinear')
                result_tensor = torch.squeeze(result_tensor, 0)
                ma = torch.max(result_tensor)
                mi = torch.min(result_tensor)
                result_tensor = (result_tensor - mi) / (ma - mi)
                im_array = (
                    (result_tensor * 255)
                    .permute(1, 2, 0)
                    .cpu()
                    .data.numpy()
                    .astype(np.uint8)
                )
                im_array = np.squeeze(im_array)
                pil_mask_im = Image.fromarray(im_array)

            # Create final images
            new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
            new_im.paste(orig_image, mask=pil_mask_im)
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_mask_im)

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        return torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiBRIA_RMBG": GeminiBRIA_RMBG,
    "GeminiBRIA_RMBG_Safe": GeminiBRIA_RMBG_Safe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiBRIA_RMBG": "BRIA RMBG (Auto-Version)",
    "GeminiBRIA_RMBG_Safe": "BRIA RMBG-1.4 (Safe Mode)",
}
