import os
from typing import Optional, Sequence

import cv2
import shadow


def gaussian_blur(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                  size: int = 5, sigma: float = 0) \
        -> cv2.typing.MatLike:
    if isinstance(image, str):
        image = cv2.imread(image)

    res = cv2.GaussianBlur(image, (size, size), sigma)

    if output is not None:
        cv2.imwrite(output, res)

    return res


def remove_shadows(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                   convolve_window_size=5, num_thresholds=3, struc_elem_size=5, exponent=1) \
        -> cv2.typing.MatLike:
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
