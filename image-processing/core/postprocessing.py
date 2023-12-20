from typing import Optional, Sequence

import cv2


def morphologie(image: str | cv2.typing.MatLike, output: Optional[str] = None,
                shape_open: int = cv2.MORPH_RECT, size_open: Sequence[int] = (3, 3),
                shape_close: int = cv2.MORPH_RECT, size_close: Sequence[int] = (3, 3),
                open_first: bool = False)\
        -> cv2.typing.MatLike:
    if isinstance(image, str):
        image = cv2.imread(image)

    open_kernel: cv2.typing.MatLike = cv2.getStructuringElement(shape_open, size_open)
    close_kernel: cv2.typing.MatLike = cv2.getStructuringElement(shape_close, size_close)

    if open_first:
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, open_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)
    else:
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, open_kernel)

    if output is not None:
        cv2.imwrite(output, image)
    return image
