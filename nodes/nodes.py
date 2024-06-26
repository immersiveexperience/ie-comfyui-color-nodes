import numpy as np
import torch
from PIL import Image
import webcolors
from webcolors._definitions import _CSS3_NAMES_TO_HEX, _CSS3_HEX_TO_NAMES


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
        color_name = webcolors.hex_to_name(hex_code, spec="css3")
    except ValueError:
        color_name = closest_color_name(hex_code)
    return color_name


def closest_color_name(hex_code):
    rgb_color = webcolors.hex_to_rgb(hex_code)

    min_colors = {}
    for name, hex_value in _CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - rgb_color[0]) ** 2
        gd = (g_c - rgb_color[1]) ** 2
        bd = (b_c - rgb_color[2]) ** 2
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


class HexColorToImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING", {"forceInput": True}),
                "image_width": ("INT", {"default": 1024}),
                "image_height": ("INT", {"default": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "hex_color_to_image"
    CATEGORY = "IE Custom Nodes"

    def hex_color_to_image(self, hex_color, image_width, image_height):
        # Create a PIL image with the specified hex color and size
        image = Image.new("RGB", (image_width, image_height), hex_color)
        image_tensor = pil2tensor(image)

        return (image_tensor,)


class HexToColorNameNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("COLOR NAME",)
    FUNCTION = "calculate_color_name"
    CATEGORY = "IE Custom Nodes"

    def calculate_color_name(self, hex_color):
        color_name = get_color_name(hex_color)
        return (color_name,)


class RandomStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"default": 10}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RANDOM STRING",)
    FUNCTION = "generate_random_string"
    CATEGORY = "IE Custom Nodes"

    def generate_random_string(self, length):
        import random
        import string

        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=length)
        )
        return (random_string,)
