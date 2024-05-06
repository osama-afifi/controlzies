import gc

import numpy as np
from PIL import Image
import torch
from scipy.signal import fftconvolve

from palette import COLOR_MAPPING, COLOR_MAPPING_


def rgb_to_hex(rgb):
    """ Converts RGB tuple to hex
    Args:
        RGB tuple
    Returns:
        hex color
    """
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]).upper()


def to_rgb(color: str) -> tuple:
    """Convert hex color to rgb.
    Args:
        color (str): hex color
    Returns:
        tuple: rgb color
    """
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


def map_colors(color: str) -> str:
    """Map color to hex value.
    Args:
        color (str): color name
    Returns:
        str: hex value
    """
    return COLOR_MAPPING[color]


def map_colors_rgb(color: tuple) -> str:
    return COLOR_MAPPING_RGB[color]


def get_color_label(color: str) -> str:
    """Get's label of a color.
    Args:
        color (str): hex color
    Returns:
        str: label string
    """
    return COLOR_MAPPING_[color]


def convolution(mask: Image.Image, size=9) -> Image:
    """Method to blur the mask
    Args:
        mask (Image): masking image
        size (int, optional): size of the blur. Defaults to 9.
    Returns:
        Image: blurred mask
    """
    mask = np.array(mask.convert("L"))
    conv = np.ones((size, size)) / size**2
    mask_blended = fftconvolve(mask, conv, 'same')
    mask_blended = mask_blended.astype(np.uint8).copy()

    border = size

    # replace borders with original values
    mask_blended[:border, :] = mask[:border, :]
    mask_blended[-border:, :] = mask[-border:, :]
    mask_blended[:, :border] = mask[:, :border]
    mask_blended[:, -border:] = mask[:, -border:]

    return Image.fromarray(mask_blended).convert("L")


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def postprocess_image_masking(inpainted: Image, image: Image,
                              mask: Image) -> Image:
    """Method to postprocess the inpainted image
    Args:
        inpainted (Image): inpainted image
        image (Image): original image
        mask (Image): mask
    Returns:
        Image: inpainted image
    """
    final_inpainted = Image.composite(inpainted.convert("RGBA"),
                                      image.convert("RGBA"), mask)
    return final_inpainted.convert("RGB")


COLOR_NAMES = list(COLOR_MAPPING.keys())
COLOR_RGB = [to_rgb(k) for k in COLOR_MAPPING_.keys()] + [(0, 0, 0),
                                                          (255, 255, 255)]
INVERSE_COLORS = {v: to_rgb(k) for k, v in COLOR_MAPPING_.items()}
COLOR_MAPPING_RGB = {to_rgb(k): v for k, v in COLOR_MAPPING_.items()}
