from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import cv2
import matplotlib.pyplot as plt
import os

import os
from PIL import Image
from tqdm import tqdm

import numpy as np

# import sys
# sys.path.append('../mask_evaluation')
# # from main import evaluate_files
# from mask_evaluation.main import evaluate_files

def eval_func(PIMG_PATH, PANN_PATH, PPRED_PATH, model):
     
     try:
        
        PIMG_files = os.listdir(PIMG_PATH)
        PIMG_files = list(filter(lambda x: x.endswith(".tif"), PIMG_files))
        PANN_files = os.listdir(PANN_PATH)
        PANN_files = list(filter(lambda x: x.endswith(".png"), PANN_files))
        PPRED_files = os.listdir(f'{PPRED_PATH}')
        PPRED_files = list(filter(lambda x: x.endswith(".tif"), PPRED_files))

        if len(PPRED_files) == 0:
            __pred_into_files(PIMG_PATH=PIMG_PATH, PIMG_files=PIMG_files, PANN_PATH=PANN_PATH, PANN_files=PANN_files, PPRED_PATH=PPRED_PATH, model=model)
            PPRED_files = os.listdir(f'{PPRED_PATH}')
            PPRED_files = list(filter(lambda x: x.endswith(".tif"), PPRED_files))

     except:
          print()

def __extract_green(image):
    img = image
    img[:,:,1] = 255
    img[:,:,2] = 255
    # print(g[0,0,:])

    for i in range(len(img[:,0,0])):
        for ii in range(len(img[0,:,0])):
            if img[i, ii, 0] != 255:
                img[i, ii, :] = [0,0,0]
    img = cv2.bitwise_not(img)
    return img

def __pred_into_files(PIMG_PATH, PIMG_files, PANN_PATH, PANN_files, PPRED_PATH, model):
    for idx, f in enumerate(PIMG_files):
        fileIdx = idx
        tIMG = cv2.imread(f'{PIMG_PATH}/{PIMG_files[fileIdx]}')
        tIMG = cv2.cvtColor(tIMG, cv2.COLOR_BGR2RGB)
        tANN = cv2.imread(f'{PANN_PATH}/{PANN_files[fileIdx]}')
        loveDa_r18_im_raw = inference_model(model, tIMG)
        loveDa_r18_raw_result = show_result_pyplot(model, np.full_like(tIMG, [255,255,255]), loveDa_r18_im_raw, with_labels=False)
        # plt.imshow(loveDa_r18_raw_result)

        predicted = __extract_green(loveDa_r18_raw_result)
        cv2.imwrite(f'{PPRED_PATH}/{f}', predicted)


    