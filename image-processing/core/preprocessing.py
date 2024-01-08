import os
import numpy as np
import cv2
from typing import Optional, Sequence
import shadow
from .tiff_handler import _load_image, _save_image


def gaussian_blur(image: str | np.ndarray, output: Optional[str] = None,
                  size: int = 5, sigma: float = 0) -> np.ndarray:
    
    image, tif_meta = _load_image(image)

    res = cv2.GaussianBlur(image, (size, size), sigma)

    if output is not None:
        _save_image(res, output, tif_meta)

    return res


def remove_shadows(image: str | np.ndarray, output: Optional[str] = None,
                   convolve_window_size=5, num_thresholds=3, struc_elem_size=5, exponent=1) \
        -> np.ndarray:
    delete_input = False
    if not isinstance(image, str):
        delete_input = True
        name = "shadow_remove_in.png"
        cv2.imwrite(name, image)
        image = name

    temp_mask = "shadow_remove_mask.png"
    shadow.shadow_detection(image, temp_mask, convolve_window_size, num_thresholds, struc_elem_size)

    delete_output = False
    if output is None:
        output = "shadow_remove_out.png"
        delete_output = True

    shadow.shadow_correction(image, temp_mask, output, exponent)
    res = cv2.imread(output)

    if delete_input:
        os.remove(image)
    os.remove(temp_mask)
    if delete_output:
        os.remove(output)

    return res


def increase_contrast(image: str | np.ndarray, output: Optional[str] = None, alpha: float = 1.5, beta: float = 0) -> np.ndarray:

    image, tif_meta = _load_image(image)

    # Apply contrast adjustment using the formula: output = alpha * input + beta
    res = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if output is not None:
        _save_image(res, output, tif_meta)

    return res


def increase_saturation(image, output=None, factor=1.5):

    image, tif_meta = _load_image(image)

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the saturation channel
    hsv_image[:, :, 1] = np.clip(
        hsv_image[:, :, 1] * factor, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    res = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if output is not None:
        _save_image(res, output, tif_meta)

    return res



