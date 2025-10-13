from .ns2d import NavierStokes2DDataset
from .ERA5wind import ERA5WindDataset
from .ERA5temperature import ERA5TemperatureDataset

_dataset_dict = {
    "NavierStokes2D": NavierStokes2DDataset,
    "ERA5wind": ERA5WindDataset,
    "ERA5temperature": ERA5TemperatureDataset,
}
