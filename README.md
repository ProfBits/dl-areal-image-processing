# dl-areal-image-processing

## Environment setup

A `environment.yml` file is provided to create a conda environment with all the required packages. To create the environment run the following command:

```bash
conda env create -f environment.yml
```

## Download data

To download the data, go into the project root directory and run the following command:

```bash
bash ./bin/download_data.sh
```

## Segment Anything

To get started you need to download the checkpoints for the models. To do so, run the following command:

```bash
bash ./bin/get_sam_models.sh
```
