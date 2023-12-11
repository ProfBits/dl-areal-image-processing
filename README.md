# dl-areal-image-processing

## Environment setup

A `environment.yml` file is provided to create a conda/mamba environment with all the required packages. To create the environment run the following command:

```bash
conda env create -f environment.yml
```

In case you want work with pip and virtualenv, a `requirements.txt` file is also provided. To create the environment run the following command:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Update environment

Please every time you add a new package to the environment, update the `environment.yml` and `requirements.txt` file accordingly.

To update the environment run the following command:

```bash
conda env update -f environment.yml
```

## Download data

To experiment with 3 chosen sample images in Munich you first need to download the data.
To download the data, go into the project root directory and run the following command:

```bash
bash ./bin/download_data.sh
```

## Segment Anything

Segment Anything needs its pretrained model checkpoints to work. These are not directly included in the repository, because of their size.
To get started you need to download the checkpoints for the models. To do so, run the following command:

```bash
bash ./bin/get_sam_models.sh
```

## Add additional info...

@Team pls specify additional info here...
