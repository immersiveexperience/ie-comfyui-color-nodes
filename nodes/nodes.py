import numpy as np
import torch
from PIL import Image
import webcolors


def pil2tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    """
    return Image.fromarray(tensor.squeeze(0).mul(255).byte().numpy().astype(np.uint8))


def get_color_name(hex_code):
    try:
        color_name = webcolors.hex_to_name(hex_code)
    except ValueError:
        color_name = closest_color_name(hex_code)
    return color_name


def closest_color_name(hex_code):
    rgb_color = webcolors.hex_to_rgb(hex_code)

    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_color.red) ** 2
        gd = (g_c - rgb_color.green) ** 2
        bd = (b_c - rgb_color.blue) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]


class AverageColorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "HEX COLOR")
    FUNCTION = "calculate_average_color"
    CATEGORY = "IE Custom Nodes"

    def calculate_average_color(self, image):
        # Ensure the input is a PIL Image, assuming 'image' input is a tensor
        input_image = tensor2pil(image)

        # Calculate the average color of the input image
        np_image = np.array(input_image)
        average_color = np.mean(np_image, axis=(0, 1))
        average_color_image = Image.new(
            "RGB", input_image.size, tuple(average_color.astype(int))
        )
        average_color_hex = "#%02x%02x%02x" % tuple(average_color.astype(int))

        # Convert the PIL image with average color back to tensor for the output
        output_image_tensor = pil2tensor(average_color_image)

        return (output_image_tensor, average_color_hex)


class ComplementaryColorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "HEX COLOR")
    FUNCTION = "calculate_complementary_color"
    CATEGORY = "IE Custom Nodes"

    def calculate_complementary_color(self, image):
        # Ensure the input is a PIL Image, assuming 'image' input is a tensor
        input_image = tensor2pil(image)

        # Calculate the complementary color of the input image
        np_image = np.array(input_image)
        complementary_color = 255 - np_image
        complementary_color_image = Image.fromarray(complementary_color)

        # Convert the PIL image with complementary color back to tensor for the output
        output_image_tensor = pil2tensor(complementary_color_image)

        return (output_image_tensor, "Complementary color")


class HexToColorNameNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("COLOR NAME",)
    FUNCTION = "calculate_color_name"
    CATEGORY = "IE Custom Nodes"

    def calculate_color_name(self, hex_color):
        color_name = get_color_name(hex_color)
        return (color_name,)
