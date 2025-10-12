# DiffSR

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

The datasets include:

- Navier-Stokes Equation from [FNO Datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-);
- Climate Data from [ECMWF Climate Data Store](https://cds.climate.copernicus.eu/);

## Usage

### Training

1. Add dataset path in the config file in the `configs` folder.

2. Run the following command to train the model in single GPU or DP mode:

```train
python main.py --config configs/ns2d/fno.yaml
```

3. Run the following command to train the model in DDP mode:

```train
torchrun --nproc_per_node=4 main_ddp.py --config /ai/gno/CODE/DiffSR/configs/ns2d/unet.yaml
```


### Code Structure

The codebase is organized as follows:

- `datasets/`: contains the dataset classes.
- `models/`: contains the model classes.
- `trainers/`: contains the model trainer classes.
- `forecastors/`: contains the forecastor base class.
- `configs/`: contains the configuration files.
- `utils/`: contains the utility functions.
- `main.py`/`main_ddp.py`: the main file to run the code.
