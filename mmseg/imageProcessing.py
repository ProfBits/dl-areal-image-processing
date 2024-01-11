# imageProcessing
import os
from PIL import Image
from tqdm.notebook import tqdm

# IMG_PATH = './data/Munich/img_dir'
# ANN_PATH = './data/Munich/ann_dir'

def cut_tif_into_patches_with_hard_boundary(i, d, patch_size):
    if not os.path.exists(d):
        os.makedirs(d)

    files = [f for f in os.listdir(i) if f.endswith('.tif')]
    for file in tqdm(files, desc='Processing images'):
        file_path = os.path.join(i, file)
        img = Image.open(file_path)
        
        width, height = img.size
        x_patches = width // patch_size
        y_patches = height // patch_size
        
        for x in range(x_patches):
            for y in range(y_patches):
                left = x* patch_size
                upper = y * patch_size
                right = (x+1) * patch_size
                lower = (y+1) * patch_size
                
                patch = img.crop((left, upper, right, lower))
                base, suffix = os.path.splitext(file)
                parts = base.rsplit('_', 1)
                patch_filename = f'{parts[0]}_{x}_{y}_{parts[1]}{suffix}'
                patch.save(os.path.join(d, patch_filename))



def cut_tif_into_patches_with_overlap(i, d, patch_size, overlap=128):
    if not os.path.exists(d):
        os.makedirs(d)
    
    files = [f for f in os.listdir(i) if f.endswith('.tif')]
    for file in tqdm(files, desc='Processing images'):
        file_path = os.path.join(i, file)
        img = Image.open(file_path)
        
        width, height = img.size
        step = patch_size - overlap  
        x_patches = (width - overlap) // step
        y_patches = (height - overlap) // step
        
        for x in range(x_patches):
            for y in range(y_patches):
                left = x * step
                upper = y * step
                right = left + patch_size
                lower = upper + patch_size
                patch = img.crop((left, upper, right, lower))
                base, suffix = os.path.splitext(file)
                parts = base.rsplit('_', 1)
                patch_filename = f'{parts[0]}_{x}_{y}_{parts[1]}{suffix}'
                patch.save(os.path.join(d, patch_filename))
                
                
# cut_tif_into_patches_with_overlap(f'{root_path}/Potsdam/5_Labels_all', f'{root_path}/Potsdam/5_Labels_all_Patched', 512)
# cut_tif_into_patches_with_overlap(f'{root_path}/Potsdam/2_Ortho_RGB', f'{root_path}/Potsdam/2_Ortho_RGB_Patched', 512)
# cut_tif_into_patches_with_hard_boundary(i=f'{IMG_PATH}', d=f'{IMG_PATH}/patches', patch_size=512)