# GPO

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Datastes

The datasets include:

- Navier-Stokes Equation from [FNO Datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-);
- Climate Data from [ECMWF Climate Data Store](https://cds.climate.copernicus.eu/);

## Usage

### Training

1. Add dataset path in the config file in the `configs` folder.

2. Run the following command to train the model:

```train
python main.py --config configs/ns2d/fno.yaml
```


### Code Structure

The codebase is organized as follows:

- `datasets/`: contains the dataset classes.
- `models/`: contains the model classes.
- `trainers/`: contrains the model builder classes.
- `procedures/`: contains the dataset procedure methods.
- `configs/`: contains the configuration files.
- `utils/`: contains the utility functions.
- `main.py`: the main file to run the code.

### Model

Write your model in `models` folder and register it in `models/__init__.py` file. 

```python
from .your_model import YourModel

ALL_MODELS = {
    ...
    'your_model': YourModel,
}
```

### Dataset

Write your dataset in directory `datasets/` and register your dataset in `datasets/__init__.py` as follows:

```python
from .your_dataset import YourDataset
```

### Trainer

Write your model builder in `trainers/` and register it in `trainers/__init__.py` as follows:

```python
from .your_model import YourModelTrainer

TRAINER_DICT = {
    ...
    'your_model': YourModelTrainer,
}
```

We provide a default trainers in `trainers/base.py` which can be used for training the model.

### Procedure

Write your dataset procedure in `procedures/` and register it in `procedures/__init__.py` as follows:

```python
from .your_dataset import YourDatasetProcedure
```

Add your dataset procedure in the `main.py` file.

```python
...
elif args.dataset == 'your_dataset':
    procedure = YourDatasetProcedure
...
```

### Config
Write your config in directory `configs/` and run the following command to train the model:

```train
python main.py --config configs/your_config.yaml
```

