from .ns2d import NavierStokes2DDataset
from .ERA5 import ERA5Dataset
from .Ocean import OceanDataset

_dataset_dict = {
    "NavierStokes2D": NavierStokes2DDataset,
    "ERA5": ERA5Dataset,
    "Ocean": OceanDataset,
}
