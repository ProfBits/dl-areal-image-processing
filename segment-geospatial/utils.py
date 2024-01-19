
import cv2
import os
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


def plot_diff_overlay(image_path, true_mask_path, predicted_mask_path, alpha=0.5, title=None, output=None):
    """
    Plot an overlay of an image with false negatives (in blue) and false positives (in red).

    Parameters
    ----------
    image_path : str
        Path to the image.
    true_mask_path : str
        Path to the true mask.
    predicted_mask_path : str
        Path to the predicted mask.
    alpha : float, optional
        Alpha value for the overlay. The default is 0.5.
    title : str, optional
        Title for the plot. The default is None.
    output : str, optional
        Path to save the plot. The default is None.

    """

    image = cv2.imread(image_path)

    # Load true and predicted masks
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
    predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    # Create binary masks
    true_binary = (true_mask > 0).astype(np.uint8)
    predicted_binary = (predicted_mask > 0).astype(np.uint8)

    # Compute false negatives (missed detections)
    false_negatives = np.logical_and(
        true_binary, 1 - predicted_binary).astype(np.uint8) * 255

    # Compute false positives (incorrect detections)
    false_positives = np.logical_and(
        1 - true_binary, predicted_binary).astype(np.uint8) * 255

    # Create three-channel color images for false negatives and false positives masks
    fn_mask = cv2.cvtColor(false_negatives, cv2.COLOR_GRAY2BGR)
    fp_mask = cv2.cvtColor(false_positives, cv2.COLOR_GRAY2BGR)

    # Convert white regions in the false negatives mask to blue tones
    fn_mask[np.where((fn_mask == [255, 255, 255]).all(axis=2))] = [
        255, 0, 0]  # (bgr)

    # Convert white regions in the false positives mask to red tones
    fp_mask[np.where((fp_mask == [255, 255, 255]).all(axis=2))] = [
        0, 0, 255]  # (bgr)

    # Combine false negatives and false positives masks
    combined_mask = cv2.addWeighted(fn_mask, 1, fp_mask, 1, 0)

    # Convert black regions in the combined mask to full color in the original image
    combined_mask[np.where((combined_mask == [0, 0, 0]).all(
        axis=2))] = image[np.where((combined_mask == [0, 0, 0]).all(axis=2))]

    # Apply the combined mask to the original image (again, to account for the brightness adjustment)
    overlay = cv2.addWeighted(image, 1 - alpha, combined_mask, alpha, 0)

    if output:
        cv2.imwrite(output, overlay)

    # Convert from BGR to RGB for matplotlib
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Display the final result using matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(overlay)
    if title:
        plt.title(title)
    plt.axis('off')
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

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_mask_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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


# Function to print geotransform and CRS information of a raster file
def print_geo_info(raster_path):
    with rasterio.open(raster_path) as src:
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")


def increase_contrast(image, output=None, alpha=1.5, beta=0):

    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image_path)

    # Apply contrast adjustment using the formula: output = alpha * input + beta
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if output:
        cv2.imwrite(output, adjusted_image)

    return adjusted_image


def increase_saturation(image, output=None, factor=1.5):

    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image_path)

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the saturation channel
    hsv_image[:, :, 1] = np.clip(
        hsv_image[:, :, 1] * factor, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    if output:
        cv2.imwrite(output, saturated_image)

    return saturated_image
