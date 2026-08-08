from enum import Enum, IntEnum

class LightModel(Enum):
    ABSORPTION = 0
    SCATTERING = 1

class SampleType(Enum):
    HELICOID = 0
    BIOPSY1 = 1
    BIOPSY2 = 2

class CriterionMetric(Enum):
    COND_NUMBER = 0
    OIF = 1
    SV_PROD = 2
    SV_SUM = 3