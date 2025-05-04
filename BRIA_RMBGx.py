import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize
import numpy as np
from transformers import AutoModelForImageSegmentation
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["1.4", "2.0"],),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "AI API"

    def load_model(self, model_version):
        # Only load if we need to (version changed or model not loaded)
        if self.model is None or self.current_version != model_version:
            if model_version == "1.4":
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "briaai/RMBG-1.4", trust_remote_code=True
                )
            elif model_version == "2.0":
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "briaai/RMBG-2.0", trust_remote_code=True
                )
            else:
                raise ValueError(f"Invalid model version: {model_version}")
            
            self.model.to(device)
            self.model.eval()
            self.current_version = model_version

    def process_image(self, model_version, image):
        self.load_model(model_version)
        processed_images = []
        processed_masks = []

        for img_tensor in image:
            orig_image = tensor2pil(img_tensor)
            w, h = orig_image.size

            # Prepare input tensor based on model version
            if model_version == "1.4":
                resized_image = resize_image(orig_image)
                im_np = np.array(resized_image)
                im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
                im_tensor = im_tensor.unsqueeze(0)
                im_tensor = im_tensor / 255.0
                im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            else:  # 2.0
                transform_image = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                im_tensor = transform_image(orig_image).unsqueeze(0)

            im_tensor = im_tensor.to(device)

            # Process image
            with torch.no_grad():
                result = self.model(im_tensor)
                
                if model_version == "1.4":
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
                    preds = result[-1].sigmoid().cpu()
                    pred = preds[0].squeeze()
                    pil_mask_im = transforms.ToPILImage()(pred)
                    pil_mask_im = pil_mask_im.resize((w, h))

            # Create final images
            new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
            new_im.paste(orig_image, mask=pil_mask_im)
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_mask_im)
            
            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        return torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0)
