from numpy import mean, maximum, max, min
from cv2 import cv2


def generate_gradcam(image, layers):
    grad = None
    data = None
    for k, v in layers[::-1]:
        if 'Convolution' in k:
            grad = v.variable_instance.g
            data = v.variable_instance.d
            break

    if grad is None or data is None:
        raise RuntimeError()

    weight_sum = calculate(data, grad)
    image = overlay(image, weight_sum)
    return image


def calculate(data, grad):
    weight = mean(grad, axis=(0, 2, 3), keepdims=True)
    weight_sum = weight * data
    weight_sum = maximum(weight_sum, 0)
    weight_sum = mean(weight_sum, axis=(0, 1))
    max_v, min_v = max(weight_sum), min(weight_sum)
    if max_v != min_v:
        weight_sum = (weight_sum - min_v) / (max_v - min_v)

    return weight_sum


def overlay(image, grad, overlay_coef=1.0):
    colormap = cv2.resize(grad, (image.shape[1], image.shape[0]))
    colormap = 255.0 * colormap / max(colormap)
    colormap = 255.0 - colormap
    colormap = colormap.aytype('uint8')

    image = colormap * overlay_coef + image
    image = 255 * image / max(image)
    image = image.astype('uint8')
    return image
