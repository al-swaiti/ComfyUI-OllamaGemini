# Import required libraries
import os
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
import re
import random
import string

# Try to import vtracer for SVG conversion
try:
    import vtracer
    VTRACER_AVAILABLE = True
    print("vtracer module found. SVG conversion will use vtracer.")
except ImportError:
    VTRACER_AVAILABLE = False
    print("Warning: vtracer module not found. SVG conversion will use basic methods.")

# Import folder_paths for ComfyUI integration
try:
    import folder_paths
except ImportError:
    print("Warning: folder_paths module not found. This may not be running in ComfyUI.")
    folder_paths = None

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def random_string(length=10):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def optimize_svg(svg_content):
    """Optimize SVG content to reduce file size using Python-only methods"""
    try:
        # Basic SVG optimization without external dependencies
        # Remove comments
        svg_content = re.sub(r'<!--[\s\S]*?-->', '', svg_content)

        # Remove unnecessary whitespace
        svg_content = re.sub(r'\s+', ' ', svg_content)
        svg_content = re.sub(r'>\s+<', '><', svg_content)

        # Remove empty attributes
        svg_content = re.sub(r'\s+\w+=""', '', svg_content)

        # Round decimal values to 2 places
        def round_decimals(match):
            try:
                value = float(match.group(0))
                return str(round(value, 2))
            except:
                return match.group(0)

        svg_content = re.sub(r'\d+\.\d+', round_decimals, svg_content)

        return svg_content
    except Exception as e:
        print(f"SVG optimization error: {str(e)}")
        return svg_content

# Create a very simple SVG preview image
def create_svg_preview(svg_string, width, height):
    """Create a very simple preview image for SVG"""
    try:
        # Create a blank image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw a border
        draw.rectangle([0, 0, width-1, height-1], outline='black')

        # Draw SVG icon in the center
        icon_size = min(width, height) // 4
        center_x = width // 2
        center_y = height // 2

        # Draw SVG text
        draw.text((center_x - 50, center_y - 60), "SVG", fill='blue')

        # Draw a simple vector graphic icon
        draw.line([(center_x - icon_size, center_y),
                  (center_x + icon_size, center_y)], fill='blue', width=2)
        draw.line([(center_x, center_y - icon_size),
                  (center_x, center_y + icon_size)], fill='blue', width=2)
        draw.ellipse([center_x - icon_size//2, center_y - icon_size//2,
                     center_x + icon_size//2, center_y + icon_size//2],
                     outline='blue', width=2)

        # Draw a message
        draw.text((center_x - 100, center_y + 50), "SVG file saved successfully", fill='green')

        return img
    except Exception as e:
        print(f"Error creating SVG preview: {str(e)}")
        # Create a simple error image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((width//4, height//2), "SVG Preview", fill='black')
        return img

def convert_with_vtracer(image, colormode, mode, filter_speckle, color_precision, corner_threshold,
                        length_threshold, max_iterations, splice_threshold, path_precision):
    """Convert image to SVG using vtracer with optimized parameters for best quality"""
    if not VTRACER_AVAILABLE:
        return None

    try:
        # Ensure image is in the right format
        if image.mode != 'RGBA':
            if image.mode != 'RGB':
                image = image.convert('RGB')
            alpha = Image.new('L', image.size, 255)
            image = Image.merge('RGBA', (*image.split(), alpha))

        # Get image data
        pixels = list(image.getdata())
        size = image.size

        # Set hierarchical mode based on colormode for best results
        hierarchical = "stacked" if colormode == "color" else "cutout"

        # Adjust parameters for better quality
        # For color images, increase color precision
        if colormode == "color":
            color_precision = max(color_precision, 6)  # At least 6 for good color reproduction

        # For spline mode, adjust corner detection
        if mode == "spline":
            corner_threshold = min(corner_threshold, 60)  # Lower threshold for better corner detection
            splice_threshold = min(splice_threshold, 45)  # Better curve splicing

        # Convert to SVG with optimized parameters
        svg_str = vtracer.convert_pixels_to_svg(
            pixels,
            size=size,
            colormode=colormode,
            hierarchical=hierarchical,  # Dynamic based on colormode
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            max_iterations=max_iterations,
            splice_threshold=splice_threshold,
            path_precision=path_precision
        )

        # Post-process SVG for better quality
        # Add viewBox attribute if missing
        if 'viewBox' not in svg_str:
            width, height = size
            svg_str = svg_str.replace('<svg ', f'<svg viewBox="0 0 {width} {height}" ', 1)

        return svg_str
    except Exception as e:
        print(f"vtracer error: {str(e)}")
        return None

# We're focusing only on vtracer for the best quality

class ConvertRasterToVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "mode": (["spline", "polygon"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 20}),
                "color_precision": ("INT", {"default": 8, "min": 1, "max": 16}),
                "corner_threshold": ("INT", {"default": 80, "min": 0, "max": 180}),
                "length_threshold": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0}),
                "max_iterations": ("INT", {"default": 15, "min": 1, "max": 50}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180}),
                "path_precision": ("INT", {"default": 5, "min": 1, "max": 10}),
                "optimize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("svg_strings",)
    FUNCTION = "convert_to_svg"

    CATEGORY = "AI API"
    DESCRIPTION = "Convert raster images to SVG vector graphics using vtracer for best quality."

    def convert_to_svg(self, image, colormode, mode, filter_speckle, color_precision,
                      corner_threshold, length_threshold, max_iterations, splice_threshold,
                      path_precision, optimize):
        svg_strings = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)

            # Check if vtracer is available
            if not VTRACER_AVAILABLE:
                # Create a simple SVG with an error message if vtracer is not available
                svg_str = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{_image.width}" height="{_image.height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="50" y="50" font-family="Arial" font-size="16" fill="red">
    Error: vtracer module not installed.
  </text>
  <text x="50" y="80" font-family="Arial" font-size="14" fill="black">
    Please install vtracer using: pip install vtracer
  </text>
  <text x="50" y="110" font-family="Arial" font-size="14" fill="black">
    Then restart ComfyUI to use this feature.
  </text>
</svg>'''
                svg_strings.append(svg_str)
                print("Error: vtracer module not installed. Please install it using: pip install vtracer")
                continue

            try:
                # Use vtracer for high-quality SVG conversion
                print("Converting image to SVG using vtracer...")
                svg_str = convert_with_vtracer(_image, colormode, mode,
                                             filter_speckle, color_precision, corner_threshold,
                                             length_threshold, max_iterations, splice_threshold,
                                             path_precision)

                # Optimize SVG if requested
                if optimize and svg_str:
                    svg_str = optimize_svg(svg_str)

                if svg_str:
                    print("Successfully converted image to SVG")
                    svg_strings.append(svg_str)
                else:
                    # Create an error SVG if conversion failed
                    error_svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{_image.width}" height="{_image.height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="50" y="50" font-family="Arial" font-size="16" fill="red">
    Error: SVG conversion failed.
  </text>
</svg>'''
                    svg_strings.append(error_svg)
                    print("Error: SVG conversion failed")
            except Exception as e:
                # Create an SVG with the error message
                error_svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{_image.width}" height="{_image.height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="50" y="50" font-family="Arial" font-size="16" fill="red">
    Error during SVG conversion: {str(e)}
  </text>
</svg>'''
                svg_strings.append(error_svg)
                print(f"Error during SVG conversion: {str(e)}")

        return (svg_strings,)

class GeminiSaveSVG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_strings": ("LIST", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_SVG"}),
                "create_preview": ("BOOLEAN", {"default": True}),
                "preview_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "preview_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "append_timestamp": ("BOOLEAN", {"default": True}),
                "custom_output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "AI API"
    DESCRIPTION = "Save SVG data to a file and generate a preview image."

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "file_path")
    OUTPUT_NODE = True
    FUNCTION = "save_svg_file"

    def generate_unique_filename(self, prefix, timestamp=False):
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp_str}.svg"
        else:
            return f"{prefix}.svg"

    def render_svg_content(self, svg_string, width, height):
        """Attempt to render the actual SVG content"""
        try:
            # Try to use Python-based SVG rendering
            # First, try to extract paths and shapes from the SVG
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*>', svg_string)
            rects = re.findall(r'<rect[^>]*>', svg_string)
            circles = re.findall(r'<circle[^>]*>', svg_string)
            ellipses = re.findall(r'<ellipse[^>]*>', svg_string)
            polygons = re.findall(r'<polygon[^>]*points="([^"]*)"[^>]*>', svg_string)

            # Extract SVG dimensions
            width_match = re.search(r'width="([\d.]+)(?:px)?"', svg_string)
            height_match = re.search(r'height="([\d.]+)(?:px)?"', svg_string)
            viewbox_match = re.search(r'viewBox="([\d\s.]+)"', svg_string)

            svg_width = width
            svg_height = height

            if width_match and height_match:
                try:
                    svg_width = float(width_match.group(1))
                    svg_height = float(height_match.group(1))
                except ValueError:
                    pass
            elif viewbox_match:
                try:
                    viewbox = viewbox_match.group(1).split()
                    if len(viewbox) >= 4:
                        svg_width = float(viewbox[2])
                        svg_height = float(viewbox[3])
                except (ValueError, IndexError):
                    pass

            # Create a blank image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)

            # Calculate scaling factors
            scale_x = width / svg_width if svg_width > 0 else 1
            scale_y = height / svg_height if svg_height > 0 else 1
            scale = min(scale_x, scale_y) * 0.9  # 90% to leave some margin

            # Draw a border
            draw.rectangle([0, 0, width-1, height-1], outline='#DDDDDD')

            # Draw the SVG content
            if paths or rects or circles or ellipses or polygons:
                # We have some SVG elements to render
                # This is a simplified rendering - a full SVG renderer would be much more complex

                # Draw paths (very simplified)
                for path in paths:
                    # Just draw a representation of the path
                    points = re.findall(r'[\d.]+,[\d.]+', path)
                    if points and len(points) > 1:
                        try:
                            # Convert points to coordinates
                            coords = []
                            for point in points[:20]:  # Limit to first 20 points
                                x, y = point.split(',')
                                x = float(x) * scale + (width - svg_width * scale) / 2
                                y = float(y) * scale + (height - svg_height * scale) / 2
                                coords.append((x, y))

                            # Draw lines between points
                            if len(coords) > 1:
                                draw.line(coords, fill='blue', width=2)
                        except (ValueError, IndexError):
                            pass

                # Draw rectangles
                for rect in rects:
                    try:
                        x_match = re.search(r'x="([\d.]+)"', rect)
                        y_match = re.search(r'y="([\d.]+)"', rect)
                        width_match = re.search(r'width="([\d.]+)"', rect)
                        height_match = re.search(r'height="([\d.]+)"', rect)

                        if x_match and y_match and width_match and height_match:
                            x = float(x_match.group(1)) * scale + (width - svg_width * scale) / 2
                            y = float(y_match.group(1)) * scale + (height - svg_height * scale) / 2
                            w = float(width_match.group(1)) * scale
                            h = float(height_match.group(1)) * scale

                            draw.rectangle([x, y, x+w, y+h], outline='blue')
                    except (ValueError, IndexError):
                        pass

                # Draw circles
                for circle in circles:
                    try:
                        cx_match = re.search(r'cx="([\d.]+)"', circle)
                        cy_match = re.search(r'cy="([\d.]+)"', circle)
                        r_match = re.search(r'r="([\d.]+)"', circle)

                        if cx_match and cy_match and r_match:
                            cx = float(cx_match.group(1)) * scale + (width - svg_width * scale) / 2
                            cy = float(cy_match.group(1)) * scale + (height - svg_height * scale) / 2
                            r = float(r_match.group(1)) * scale

                            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline='blue')
                    except (ValueError, IndexError):
                        pass

                # Draw a message
                draw.text((width//2 - 100, height - 30),
                         "SVG Preview (simplified rendering)", fill='green')

                return img
            else:
                # No recognizable SVG elements found, fall back to info display
                return None
        except Exception as e:
            print(f"Error rendering SVG content: {str(e)}")
            return None

    def svg_to_image(self, svg_string, width, height, show_actual_svg=True):
        """Create a preview image for SVG"""
        try:
            # Try to render the actual SVG content if requested
            if show_actual_svg:
                rendered_img = self.render_svg_content(svg_string, width, height)
                if rendered_img:
                    return rendered_img

            # Extract dimensions from SVG
            svg_width = width
            svg_height = height

            # Try to extract actual dimensions from SVG
            width_match = re.search(r'width="([\d.]+)(?:px)?"', svg_string)
            height_match = re.search(r'height="([\d.]+)(?:px)?"', svg_string)
            viewbox_match = re.search(r'viewBox="([\d\s.]+)"', svg_string)

            if width_match and height_match:
                try:
                    svg_width = float(width_match.group(1))
                    svg_height = float(height_match.group(1))
                except ValueError:
                    pass  # Use default dimensions
            elif viewbox_match:
                try:
                    viewbox = viewbox_match.group(1).split()
                    if len(viewbox) >= 4:
                        svg_width = float(viewbox[2])
                        svg_height = float(viewbox[3])
                except (ValueError, IndexError):
                    pass

            # Create a blank image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)

            # Draw a border
            draw.rectangle([0, 0, width-1, height-1], outline='black')

            # Draw SVG icon in the center
            icon_size = min(width, height) // 4
            center_x = width // 2
            center_y = height // 2

            # Draw SVG text
            draw.text((center_x - 50, center_y - 60), "SVG", fill='blue')

            # Draw a simple vector graphic icon
            draw.line([(center_x - icon_size, center_y),
                      (center_x + icon_size, center_y)], fill='blue', width=2)
            draw.line([(center_x, center_y - icon_size),
                      (center_x, center_y + icon_size)], fill='blue', width=2)
            draw.ellipse([center_x - icon_size//2, center_y - icon_size//2,
                         center_x + icon_size//2, center_y + icon_size//2],
                         outline='blue', width=2)

            # Draw dimensions
            draw.text((center_x - 70, center_y + 40),
                     f"Size: {int(svg_width)} × {int(svg_height)}", fill='black')

            # Draw file size
            file_size_kb = len(svg_string) / 1024
            draw.text((center_x - 70, center_y + 60),
                     f"File size: {file_size_kb:.1f} KB", fill='black')

            # Draw a message
            draw.text((width//2 - 100, height - 40),
                     "SVG file saved successfully", fill='green')

            return img
        except Exception as e:
            print(f"Error creating SVG preview: {str(e)}")
            # Create a simple error image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((width//4, height//2), f"SVG Preview Error: {str(e)}", fill='red')
            return img

    def save_svg_file(self, svg_strings, filename_prefix="ComfyUI_SVG", create_preview=True,
                     preview_width=512, preview_height=512,
                     append_timestamp=True, custom_output_path=""):
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)

        saved_paths = []
        preview_images = []

        for index, svg_string in enumerate(svg_strings):
            # Generate a unique filename
            unique_filename = self.generate_unique_filename(f"{filename_prefix}_{index}", append_timestamp)
            final_filepath = os.path.join(output_path, unique_filename)

            # Save the SVG file
            try:
                with open(final_filepath, "w", encoding='utf-8') as svg_file:
                    svg_file.write(svg_string)
                print(f"SVG saved to: {final_filepath}")
                saved_paths.append(final_filepath)
            except Exception as e:
                print(f"Error saving SVG file: {str(e)}")
                saved_paths.append(f"Error: {str(e)}")

            # Create preview image if requested
            if create_preview:
                preview_img = create_svg_preview(svg_string, preview_width, preview_height)
                preview_tensor = pil2tensor(preview_img)
                preview_images.append(preview_tensor)

        # Combine all preview images into a single tensor
        if preview_images:
            preview_tensor = torch.cat(preview_images, dim=0)
        else:
            # Create a blank image if no previews were generated
            blank_img = Image.new('RGB', (preview_width, preview_height), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(blank_img)
            draw.text((preview_width//4, preview_height//2), "No SVG files were processed", fill='black')
            preview_tensor = pil2tensor(blank_img)

        # Return the preview image tensor and the list of saved file paths
        return (preview_tensor, "\n".join(saved_paths))

class GeminiSVGPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_strings": ("LIST", {"forceInput": True}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview_svg"

    CATEGORY = "AI API"
    DESCRIPTION = "Preview SVG data as an image with improved rendering."

    def render_svg_content(self, svg_string, width, height):
        """Attempt to render the actual SVG content"""
        try:
            # Try to use Python-based SVG rendering
            # First, try to extract paths and shapes from the SVG
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*>', svg_string)
            rects = re.findall(r'<rect[^>]*>', svg_string)
            circles = re.findall(r'<circle[^>]*>', svg_string)
            ellipses = re.findall(r'<ellipse[^>]*>', svg_string)
            polygons = re.findall(r'<polygon[^>]*points="([^"]*)"[^>]*>', svg_string)

            # Extract SVG dimensions
            width_match = re.search(r'width="([\d.]+)(?:px)?"', svg_string)
            height_match = re.search(r'height="([\d.]+)(?:px)?"', svg_string)
            viewbox_match = re.search(r'viewBox="([\d\s.]+)"', svg_string)

            svg_width = width
            svg_height = height

            if width_match and height_match:
                try:
                    svg_width = float(width_match.group(1))
                    svg_height = float(height_match.group(1))
                except ValueError:
                    pass
            elif viewbox_match:
                try:
                    viewbox = viewbox_match.group(1).split()
                    if len(viewbox) >= 4:
                        svg_width = float(viewbox[2])
                        svg_height = float(viewbox[3])
                except (ValueError, IndexError):
                    pass

            # Create a blank image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)

            # Calculate scaling factors
            scale_x = width / svg_width if svg_width > 0 else 1
            scale_y = height / svg_height if svg_height > 0 else 1
            scale = min(scale_x, scale_y) * 0.9  # 90% to leave some margin

            # Draw a border
            draw.rectangle([0, 0, width-1, height-1], outline='#DDDDDD')

            # Draw the SVG content
            if paths or rects or circles or ellipses or polygons:
                # We have some SVG elements to render
                # This is a simplified rendering - a full SVG renderer would be much more complex

                # Draw paths (very simplified)
                for path in paths:
                    # Just draw a representation of the path
                    points = re.findall(r'[\d.]+,[\d.]+', path)
                    if points and len(points) > 1:
                        try:
                            # Convert points to coordinates
                            coords = []
                            for point in points[:20]:  # Limit to first 20 points
                                x, y = point.split(',')
                                x = float(x) * scale + (width - svg_width * scale) / 2
                                y = float(y) * scale + (height - svg_height * scale) / 2
                                coords.append((x, y))

                            # Draw lines between points
                            if len(coords) > 1:
                                draw.line(coords, fill='blue', width=2)
                        except (ValueError, IndexError):
                            pass

                # Draw rectangles
                for rect in rects:
                    try:
                        x_match = re.search(r'x="([\d.]+)"', rect)
                        y_match = re.search(r'y="([\d.]+)"', rect)
                        width_match = re.search(r'width="([\d.]+)"', rect)
                        height_match = re.search(r'height="([\d.]+)"', rect)

                        if x_match and y_match and width_match and height_match:
                            x = float(x_match.group(1)) * scale + (width - svg_width * scale) / 2
                            y = float(y_match.group(1)) * scale + (height - svg_height * scale) / 2
                            w = float(width_match.group(1)) * scale
                            h = float(height_match.group(1)) * scale

                            draw.rectangle([x, y, x+w, y+h], outline='blue')
                    except (ValueError, IndexError):
                        pass

                # Draw circles
                for circle in circles:
                    try:
                        cx_match = re.search(r'cx="([\d.]+)"', circle)
                        cy_match = re.search(r'cy="([\d.]+)"', circle)
                        r_match = re.search(r'r="([\d.]+)"', circle)

                        if cx_match and cy_match and r_match:
                            cx = float(cx_match.group(1)) * scale + (width - svg_width * scale) / 2
                            cy = float(cy_match.group(1)) * scale + (height - svg_height * scale) / 2
                            r = float(r_match.group(1)) * scale

                            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline='blue')
                    except (ValueError, IndexError):
                        pass

                # Draw a message
                draw.text((width//2 - 100, height - 30),
                         "SVG Preview (simplified rendering)", fill='green')

                return img
            else:
                # No recognizable SVG elements found, fall back to info display
                return None
        except Exception as e:
            print(f"Error rendering SVG content: {str(e)}")
            return None

    def svg_to_image(self, svg_string, width, height, show_actual_svg=True):
        """Create a preview image for SVG"""
        try:
            # Try to render the actual SVG content if requested
            if show_actual_svg:
                rendered_img = self.render_svg_content(svg_string, width, height)
                if rendered_img:
                    return rendered_img

            # Extract dimensions from SVG
            svg_width = width
            svg_height = height

            # Try to extract actual dimensions from SVG
            width_match = re.search(r'width="([\d.]+)(?:px)?"', svg_string)
            height_match = re.search(r'height="([\d.]+)(?:px)?"', svg_string)
            viewbox_match = re.search(r'viewBox="([\d\s.]+)"', svg_string)

            if width_match and height_match:
                try:
                    svg_width = float(width_match.group(1))
                    svg_height = float(height_match.group(1))
                except ValueError:
                    pass  # Use default dimensions
            elif viewbox_match:
                try:
                    viewbox = viewbox_match.group(1).split()
                    if len(viewbox) >= 4:
                        svg_width = float(viewbox[2])
                        svg_height = float(viewbox[3])
                except (ValueError, IndexError):
                    pass

            # Create a blank image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)

            # Draw a border
            draw.rectangle([0, 0, width-1, height-1], outline='black')

            # Draw SVG icon in the center
            icon_size = min(width, height) // 4
            center_x = width // 2
            center_y = height // 2

            # Draw SVG text
            draw.text((center_x - 50, center_y - 60), "SVG", fill='blue')

            # Draw a simple vector graphic icon
            draw.line([(center_x - icon_size, center_y),
                      (center_x + icon_size, center_y)], fill='blue', width=2)
            draw.line([(center_x, center_y - icon_size),
                      (center_x, center_y + icon_size)], fill='blue', width=2)
            draw.ellipse([center_x - icon_size//2, center_y - icon_size//2,
                         center_x + icon_size//2, center_y + icon_size//2],
                         outline='blue', width=2)

            # Draw dimensions
            draw.text((center_x - 70, center_y + 40),
                     f"Size: {int(svg_width)} × {int(svg_height)}", fill='black')

            # Draw file size
            file_size_kb = len(svg_string) / 1024
            draw.text((center_x - 70, center_y + 60),
                     f"File size: {file_size_kb:.1f} KB", fill='black')

            # Draw a message
            draw.text((width//2 - 100, height - 40),
                     "SVG file saved successfully", fill='green')

            return img
        except Exception as e:
            print(f"Error creating SVG preview: {str(e)}")
            # Create a simple error image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((width//4, height//2), f"SVG Preview Error: {str(e)}", fill='red')
            return img

    def preview_svg(self, svg_strings, width=512, height=512):
        preview_images = []

        for svg_string in svg_strings:
            # Use the global create_svg_preview function
            preview_img = create_svg_preview(svg_string, width, height)
            preview_tensor = pil2tensor(preview_img)
            preview_images.append(preview_tensor)

        # Combine all preview images into a single tensor
        if preview_images:
            preview_tensor = torch.cat(preview_images, dim=0)
        else:
            # Create a blank image if no previews were generated
            blank_img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(blank_img)
            draw.text((width//4, height//2), "No SVG data to preview", fill='black')
            preview_tensor = pil2tensor(blank_img)

        return (preview_tensor,)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "ConvertRasterToVector": ConvertRasterToVector,
    "GeminiSaveSVG": GeminiSaveSVG,
    "GeminiSVGPreview": GeminiSVGPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertRasterToVector": "Convert Image to SVG",
    "GeminiSaveSVG": "Save SVG File",
    "GeminiSVGPreview": "Preview SVG"
}
