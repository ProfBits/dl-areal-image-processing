import shadow.Shadow_Detection as Shadow
import v1.green_detector


def process_image(image_path, save_path="07_no_shadows_greens.png"):
    Shadow.shadow_detection(image_path, "05_shadow_mask.png")
    Shadow.shadow_correction(image_path, "05_shadow_mask.png", "06_shadow_corrected.png")
    v1.green_detector.process_image("06_shadow_corrected.png", save_path)
