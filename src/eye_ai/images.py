from os import path, remove
from os.path import isfile
from enum import IntEnum, auto
from logging import getLogger

from numpy import uint8, fromfile, ndarray, transpose, reshape
from cv2 import cv2

logger = getLogger(__name__)


def read_image(file, flags=cv2.IMREAD_COLOR, dtype=uint8):
    try:
        n = fromfile(file, dtype)
        image = cv2.imdecode(n, flags)
        return image
    except Exception as e:
        logger.error(str(e))
        return None


def write_image(file, image: ndarray, params=None):
    try:
        ext = path.splitext(file)[1]
        result, n = cv2.imencode(ext, image, params)
        if result:
            with open(file, mode='w+b') as f:
                n.tofile(f)
    except Exception as e:
        if isfile(file):
            remove(file)
        logger.error(str(e))
        message = _('The image could not be saved')
        raise RuntimeError(f'{message}({e})')


class ImageColor(IntEnum):
    Gray = auto()
    Color = auto()


def adjust_shape(image, input_shape):
    _, color, h, w = input_shape
    image_length = len(image.shape)
    if color == 1:
        if image_length == 2:
            pass
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    elif color == 3:
        if image_length == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            pass
        image = transpose(image, (2, 0, 1))
    else:
        raise RuntimeError()
    image = reshape(image, (1, color, h, w))
    return image
