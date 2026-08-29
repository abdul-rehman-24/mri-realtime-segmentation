
import numpy as np
import cv2

def normalize_mri(image, lower_percentile=1, upper_percentile=99):
    image = image.astype(np.float32)
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    image = np.clip(image, lower, upper)
    if upper - lower > 0:
        image = (image - lower) / (upper - lower)
    else:
        image = np.zeros_like(image)
    return image

def pad_to_square(image, pad_value=0):
    h, w = image.shape
    if h == w:
        return image
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    if h < w:
        image = np.pad(image, ((pad1, pad2), (0, 0)), mode='constant', constant_values=pad_value)
    else:
        image = np.pad(image, ((0, 0), (pad1, pad2)), mode='constant', constant_values=pad_value)
    return image

def resize_mri(image, target_size=256):
    image = pad_to_square(image, pad_value=0)
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def resize_mask(mask, target_size=256):
    mask = pad_to_square(mask, pad_value=0)
    return cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
