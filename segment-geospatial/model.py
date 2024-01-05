import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from samgeo import split_raster, merge_rasters
from samgeo.text_sam import LangSAM
from utils import plot_overlay, create_empty_mask, increase_contrast, increase_saturation


# define paths
TRAIN_PATH = '../data/Munich/2023/raw'
PREDICTED_PATH = '../data/Munich/2023/prediction'
MODEL_TYPE = "vit_h" # "vit_h" | "vit_l"
# subfolder to store predictions
SUBFOLDER = "samgeo_batches_v2"

# read all files
train_files = os.listdir(TRAIN_PATH)
train_files = list(filter(lambda x: x.endswith(".tif"), train_files))

if MODEL_TYPE == "vit_l":
    SAM_CHECKPOINT = "../segment-anything/checkpoints/sam_vit_l_0b3195.pth"
else: # vit_h
    SAM_CHECKPOINT = "../segment-anything/checkpoints/sam_vit_h_4b8939.pth"


sam = LangSAM(model_type=MODEL_TYPE, checkpoint=SAM_CHECKPOINT)


box_threshold = 0.25
text_threshold = 0.5
text_prompt = "tree . lawn . gras"
tile_size = 512
overlap = 64


result_path = os.path.join(PREDICTED_PATH, SUBFOLDER)

# create prediction directory
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
# create info file with parameters
with open(os.path.join(result_path, "info.txt"), "w") as f:
    f.write("box_threshold: {}\n".format(box_threshold))
    f.write("text_threshold: {}\n".format(text_threshold))
    f.write("text_prompt: {}\n".format(text_prompt))
    f.write("model_type: {}\n".format(MODEL_TYPE))
    f.write("tile_size: {}\n".format(tile_size))
    f.write("overlap: {}\n".format(overlap))
    
# tmp folder for batch segmentation
work_dir = "tmp"
tiles_dir = f"{work_dir}/tiles"
mask_dir = f"{work_dir}/masks"


for file_name in train_files:
    
    img_name = file_name.rstrip('.tif')
    img_path = os.path.join(TRAIN_PATH, file_name)
    
    # empty tmp directory
    os.system(f"rm -rf {work_dir}/*")
    
    # split images into tiles
    split_raster(img_path, out_dir=tiles_dir, tile_size=tile_size, overlap=overlap)
    
    # sam.batch_predict written on my own to fix not creating empty masks
    if isinstance(tiles_dir, str):
        all_files = os.listdir(tiles_dir)
        images = [os.path.join(tiles_dir, file) for file in all_files if file.endswith(".tif")]
        images.sort()

    for i, image in enumerate(images):
        basename = os.path.splitext(os.path.basename(image))[0]

        output = os.path.join(mask_dir, f"{basename}_mask.tif")
        res = sam.predict(
            image,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            output=output,
            return_results=True
        )
        
        # create empty mask because samgeo does not create empty masks
        if res is None:
            create_empty_mask(image, output)
        

    # merge tiles into one mask
    mask_path = os.path.join(result_path, img_name + '_predicted.tif')
    merge_rasters(mask_dir, mask_path)
    
    

print(f"Done! Results saved to: {result_path}") 
