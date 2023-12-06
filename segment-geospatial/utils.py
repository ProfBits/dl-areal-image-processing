
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio


def plot_overlay(image_path, mask_math, alpha=0.5, title=None, output=None):
    """
    Plot an overlay of an image and a mask.

    Parameters
    ----------
    image_path : str
        Path to the image.
    mask_math : str
        Path to the mask.
    alpha : float, optional
        Alpha value for the overlay. The default is 0.5.
    title : str, optional
        Title for the plot. The default is None.
    output : str, optional
        Path to save the plot. The default is None.

    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_math)

    # Convert black regions in the mask to full color in the original image
    mask[np.where((mask == [0, 0, 0]).all(axis=2))
         ] = image[np.where((mask == [0, 0, 0]).all(axis=2))]
    # Convert white regions in the mask to green tones in the overlay
    mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

    # Apply the mask
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

    if output:
        cv2.imwrite(output, overlay)

    # convert from BGR to RGB for matplotlib
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Display the result using matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(overlay)
    if title:
        plt.title(title)
    plt.axis('off')
    # if output:
    #     plt.savefig(output, bbox_inches='tight')
    plt.show()


def create_empty_mask(image_path, output_mask_path):
    """
    Create an empty mask with the same dimensions as the image.

    Parameters
    ----------
    image_path : str
        Path to the image.
    output_mask_path : str
        Path to save the mask.
    """

    # Read the image using rasterio
    with rasterio.open(image_path) as src:
        # Get the geotransform and other relevant information
        crs = src.crs
        transform = src.transform
        dtype = np.uint8

        array = np.zeros((src.height, src.width), dtype=dtype)

        metadata = {
            'driver': 'GTiff',
            'height': src.height,
            'width': src.width,
            'count': 1,  # Number of bands (1 for grayscale)
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'deflate',
        }

        # Create an empty mask using rasterio
        with rasterio.open(output_mask_path, 'w', **metadata) as dst:
            # Write an empty array to the mask
            dst.write(array, 1)
