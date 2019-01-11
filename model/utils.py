"""model utils"""

from PIL import Image
import numpy as np
import keras.backend as K


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


def serial_apply(*funcs):
    """serial apply a list of functions

    funcs: a list of functions
    Returns: the function after serially applied
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


def sigmoid_focal_loss(_sentinel=None, y=None, y_true=None, gama=0.0):
    """ Calculate focal loss, element wise focal loss

    param _sentinel: Used to prevent positional parameters. Internal, do not use.
    param y: tensor, the predict, value should be in (0, 1), shape=(N1, N2, ..., 1)
    param y_true: tensor, ground truth, value should be 0 or 1, has the same shape with y
    param gama: float, focal factor
    return: tensor, focal loss, has the same shape with y and y_true
    """
    y_true = K.cast(y_true, dtype=K.dtype(y))
    clipped_y = K.clip(y, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.pow(1.0 - clipped_y, gama) * K.log(clipped_y) + \
        (1.0 - y_true) * K.pow(clipped_y, gama) * K.log(1 - clipped_y)

    return -loss


