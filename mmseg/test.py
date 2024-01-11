from mmseg.apis import MMSegInferencer
import cv2
import sys

## Load models into memory
# inferencer = MMSegInferencer(model='pspnet_r18-d8_4xb4-80k_loveda-512x512')
# # Inference
# inferencer('demopics/32686_5337_masked.tif', show=True)

# # Load models into memory
# inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb4-80k_loveda-512x512')
# # Inference
# inferencer('demopics/32686_5337.tif', show=True)

# # Load models into memory
# inferencer = MMSegInferencer(model='fcn_hr18s_4xb4-80k_loveda-512x512')
# # Inference
# inferencer('demopics/32686_5337.tif', show=True)

# Load models into memory
inferencer = MMSegInferencer(model='fcn_hr18s_4xb4-80k_vaihingen-512x512')
# Inference
inferencer('mmseg/demopics/32686_5337.tif', show=True)
