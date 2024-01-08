from mmseg.apis import MMSegInferencer
# Load models into memory
inferencer = MMSegInferencer(model='pspnet_r18-d8_4xb4-80k_loveda-512x512')
# Inference
inferencer('demopics/32691_5335_masked.tif', show=True)