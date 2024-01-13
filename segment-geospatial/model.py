# run inside segment-geospatial folder

import os
import sys
import time
from samgeo import split_raster, merge_rasters
from samgeo.text_sam import LangSAM
from utils import create_empty_mask
# evaluation package
sys.path.append('../mask_evaluation')
from main import print_metrics, evaluate_files
# image preprocessing package
sys.path.append('../image-processing')
from core.preprocessing import gaussian_blur, increase_contrast, increase_saturation, remove_shadows
from cut_houses.cut_houses import cut_houses



####### SET PARAMETERS  #######
TRAIN_PATH = '../data/Munich/2023/raw'
PREDICTED_PATH = '../data/Munich/2023/prediction'
MODEL_TYPE = "vit_h" # "vit_h" | "vit_l"
SUBFOLDER = "samgeo_batches_v7" # subfolder to store predictions
#######   FINE TUNING   #######
box_threshold = 0.25
text_threshold = 0.5
text_prompt = "tree . lawn . gras"
tile_size = 256
overlap = 64
img_processing = []
# img_processing = ["remove_shadows", "gaussian_blur", "increase_contrast", "increase_saturation", "cut_houses"]
cut_houses_from_mask = True 
###############################



### Create LangSAM Model ####
if MODEL_TYPE == "vit_l":
    SAM_CHECKPOINT = "../segment-anything/checkpoints/sam_vit_l_0b3195.pth"
else: # vit_h
    SAM_CHECKPOINT = "../segment-anything/checkpoints/sam_vit_h_4b8939.pth"

lang_sam = LangSAM(model_type=MODEL_TYPE, checkpoint=SAM_CHECKPOINT)


# read all files
train_files = os.listdir(TRAIN_PATH)
train_files = list(filter(lambda x: x.endswith(".tif"), train_files))

result_path = os.path.join(PREDICTED_PATH, SUBFOLDER)

# create prediction directory
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
# create info file with parameters
info_file = os.path.join(result_path, "info.txt")
with open(info_file, "w") as f:
    f.write("box_threshold: {}\n".format(box_threshold))
    f.write("text_threshold: {}\n".format(text_threshold))
    f.write("text_prompt: {}\n".format(text_prompt))
    f.write("model_type: {}\n".format(MODEL_TYPE))
    f.write("tile_size: {}\n".format(tile_size))
    f.write("overlap: {}\n".format(overlap))
    f.write("preprocessing: {}\n".format(" | ".join(img_processing)))  # write processing steps to info file
    if cut_houses_from_mask: 
        f.write("Houses were cut from predicted mask\n")
    
# tmp folder for batch segmentation
work_dir = "tmp"
tiles_dir = f"{work_dir}/tiles"
mask_dir = f"{work_dir}/masks"

# timing the prediction
start_time = time.time()

for file_name in train_files:
    
    img_name = file_name.rstrip('.tif')
    img_path = os.path.join(TRAIN_PATH, file_name)

    # empty tmp directory
    os.system(f"rm -rf {work_dir}/*")

    img_modified = "tmp/image_preprocessed.tif"
    os.system(f'cp "{img_path}" "{img_modified}"')

    # Preprocessing Pipeline
    for step in img_processing:
        if step == "remove_shadows":
            remove_shadows(img_modified, output=img_modified)
        elif step == "increase_contrast":
            increase_contrast(img_modified, output=img_modified, alpha=1.5, beta=-50)
        elif step == "increase_saturation":
            increase_saturation(img_modified, output=img_modified, factor=1.5)
        elif step == "gaussian_blur":
            gaussian_blur(img_modified, output=img_modified)
        elif step == "cut_houses":
            cut_houses(img_modified, output=img_modified)    
    
    # split images into tiles
    split_raster(img_modified, out_dir=tiles_dir, tile_size=tile_size, overlap=overlap)
    # create tmp/masks
    os.makedirs(mask_dir)
    
    lang_sam.predict_batch(
        images=tiles_dir,
        out_dir=mask_dir,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        merge=True,
    )
    mask_path = os.path.join(result_path, img_name + '_predicted.tif')
    os.system(f"mv {mask_dir}/merged.tif {mask_path}")
    
    if cut_houses_from_mask:
        cut_houses(mask_path, output=mask_path)
    print(f"Prediction finished for file: {file_name}")
    

print(f"Done! Results saved to: {result_path}") 

# End time for predictions
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_hours = elapsed_time / 3600
print(f"Total prediction time: {elapsed_hours:.2f} hours")

### Evaluation ###
masks = []

# create list of masks (base_mask, predicted_mask)
for file_name in train_files:
    img_name = file_name.rstrip('.tif')
    mask_path = os.path.join(TRAIN_PATH, img_name + '_label.png')
    predicted_path = os.path.join(result_path, img_name + '_predicted.tif')
    masks.append((predicted_path, mask_path))

evaluation = evaluate_files(masks)

# Call the print_metrics function with the file parameter
with open(info_file, "a") as metrics_file:
    metrics_file.write("\n")
    print_metrics(evaluation, file=metrics_file)
    metrics_file.write(f"\nPrediction time: {elapsed_hours:.2f} hours\n")

# print to command line
print_metrics(evaluation)
