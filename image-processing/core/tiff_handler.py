import numpy as np
import cv2
from typing import Optional
import rasterio


def _load_image(image: str | np.ndarray) -> tuple[np.ndarray, Optional[dict]]:
    meta = None
    if isinstance(image, str):
        if image.endswith(".tif"):
            with rasterio.open(image) as src:
                meta = src.meta.copy()
                # Read with Rasterio and transpose to OpenCV format:
                # from (bands, height, width) to (height, width, bands)
                image = src.read().transpose(1, 2, 0)
        else:
            image = cv2.imread(image)

    return image, meta


def _save_image(res: np.ndarray, output: str, tif_meta: Optional[dict]) -> None:

    if tif_meta is not None:  # TIFF file
        # Convert from BGR (OpenCV) to RGB
        if tif_meta["count"] == 3:
            tif_res = res[..., ::-1]  # Change channel ordering
            # Convert blurred result back to Rasterio format: (bands, height, width)
            tif_res = res.transpose(2, 0, 1)
        else:
            tif_res = res[np.newaxis, ...]

        with rasterio.open(output, 'w', **tif_meta) as dst:
            dst.write(tif_res, indexes=list(range(1, tif_meta['count'] + 1)))
    else:
        cv2.imwrite(output, res)
