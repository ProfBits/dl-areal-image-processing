# U-Net
Area segmentation with [U-Net](https://paperswithcode.com/method/u-net), a convolutional neural network (CNN) by training a model from labeled masks.
This project uses the Potsdam dataset of the isprs.

The project is split into two jupyter notebooks, which are highly customizable. 

- [image-preprocessing.ipynb](image-preprocessing.ipynb): Unzips, renames, cuts and stores images and masks of the Potsdam dataset.
- [uNet-model.ipynb](uNet-model.ipynb): Builds the Model, trains and evaluates.


## Installation:

1. Download the Potsdam (13GB) dataset from [isprs](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx) and place into unet/ folder.

```bash
# Navigate to root folder of unet
cd unet
# Create new virtual environment
python3 -m venv unet
# Activate
source unet/bin/activate
# Install required packages
python3 -m pip install -r requirements.txt
# Install kernel with custom virtual environment
python3 -m ipykernel install --user --name=unet --display-name="Python (myenv)"
# Start Jupyter Notebook
jupyter notebook
```

## Usage:
Start by preprocessing the dataset by using the [image-preprocessing.ipynb](image-preprocessing.ipynb).
For the training of the model a CUDA - GPU is highly advised.
