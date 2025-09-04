from .gpo import GPO
from .gs import GaussianField
from .dgpo import DynamicGPO
from .fno import FNO1d, FNO2d


ALL_MODELS = {
    "GPO": GPO,
    "DGPO": DynamicGPO,
    "GaussianField": GaussianField,
    "FNO1d": FNO1d,
    "FNO2d": FNO2d,
}