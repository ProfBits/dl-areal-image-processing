from enum import Enum
from typing import Optional

import cv2
import numpy as np

R = 0
G = 1
B = 2
H = 0
S = 1
V = 2


def rgb_detection(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                  red_weight=0.5, rb_offset=0.0,
                  green_weight=1.0, sub_weight=1.0, sub_offset=0.0)\
        -> cv2.typing.MatLike:
    if image is str:
        image = cv2.imread(image)

    (b, g, r) = cv2.split(image)
    rb = cv2.addWeighted(r, red_weight, b, 1 - red_weight, rb_offset)
    res = cv2.addWeighted(g, green_weight, rb, -sub_weight, sub_offset)

    if output is not None:
        cv2.imwrite(output, res)

    return res


class Limit(Enum):
    H_MIN = 0
    H_MAX = 0
    S_MIN = 0
    S_MAX = 0
    V_MIN = 0
    V_MAX = 0


def __check_bounds(image: cv2.typing.MatLike, lower: int, upper: int) -> np.ndarray:
    lower_bound = (np.ones(image.shape) * lower) <= image
    upper_bound = image <= (np.ones(image.shape) * upper)
    return np.logical_and(lower_bound, upper_bound)


def hsv_detection(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                  limits: list[dict[Limit, int]] = None)\
        -> cv2.typing.MatLike:
    if limits is None:
        raise Exception("Limits need to be configured")

    if image is str:
        image = cv2.imread(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv)

    res = np.zeros(h.shape)
    for limit in limits:
        h_limit = __check_bounds(h, limit[Limit.H_MIN], limit[Limit.H_MAX])
        s_limit = __check_bounds(s, limit[Limit.S_MIN], limit[Limit.S_MAX])
        v_limit = __check_bounds(v, limit[Limit.V_MIN], limit[Limit.V_MAX])
        res = np.logical_or(res, np.logical_and(h_limit, s_limit, v_limit)) * 1

    if output is not None:
        cv2.imwrite(output, res)

    return res


def histogram_adjustment(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                         slope: int = 1, offset: int = 0)\
        -> cv2.typing.MatLike:
    if isinstance(image, str):
        image = cv2.imread(image)

    res = np.add(np.multiply(image, slope), offset)
    res = np.clip(res, 0, 255)
    res = cv2.convertScaleAbs(res)

    if output is not None:
        cv2.imwrite(output, res)
    return res


def binarisation(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                 threshold: int = 127)\
        -> cv2.typing.MatLike:
    if isinstance(image, str):
        image = cv2.imread(image)

    _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    if output is not None:
        cv2.imwrite(output, binary_mask)
    return binary_mask
