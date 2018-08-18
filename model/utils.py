"""model utils"""

from PIL import Image
import numpy as np


def resize_image(image, size):
    """Resize image with aspect ratio kept

    image: Image, input image
    size: tuple of int, (w, h) target size
    Return: resized image
    """
    iw, ih = image.size
    w, h = size
    ratio = min(w / iw, h / ih)
    new_w = int(iw * ratio)
    new_h = int(ih * ratio)

    image = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))

    return new_image


def squeeze(*funcs):
    """squeeze a list of functions to one

    funcs: a list of functions
    Returns: squeezed function
    """
    def squeezed(*args, **kwargs):
        first_func = funcs[0]
        remain_funcs = funcs[1:]
        result = first_func(*args, **kwargs)
        for f in remain_funcs:
            result = f(result)
        return result
    if funcs:
        return squeezed
    else:
        raise ValueError('funcs can not be empty!')


def rand(a=0, b=1):
    """Sample random value between a and b

    a: random low bound
    b: random high bound
    return: random value [a, b]
    """
    return np.random.rand() * (b - a) + a
