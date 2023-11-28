import cv2
import shadow
import os

def process_image(image_path: str, save_path: str="13_automated_noise_reduced_x2.png", reduce_shadows: bool = True):
    image: cv2.typing.MatLike

    if reduce_shadows:
        mask = "shadow_mask.temp.png"
        shadow.shadow_detection(image_path, mask)
        corrected = "shadow_corrected.temp.png"
        shadow.shadow_correction(image_path, mask, corrected)
        os.remove(mask)
        image = cv2.imread(corrected)
        os.remove(corrected)
    else:
        image = cv2.imread(image_path)

    blured = cv2.GaussianBlur(image, (5, 5), 0)
    (b, g, r) = cv2.split(blured)
    rb = cv2.addWeighted(r, 0.5, b, 0.5, 0)
    greens = cv2.subtract(g, rb)
    equalized = cv2.equalizeHist(greens)
    _, raw_binary_mask = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)

    kernel: cv2.typing.MatLike = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(raw_binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(save_path, binary_mask)




