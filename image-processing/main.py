import v1.green_detector
import v2.shadow_corrected_green_detector
import v3.noise_reduction_green_detector

if __name__ == '__main__':
    #v1.green_detector.process_image("00_input.tif")
    #v2.shadow_corrected_green_detector.process_image("00_input.tif")
    v3.noise_reduction_green_detector.process_image("00_input.tif")
