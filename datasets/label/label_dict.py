from .label_lidc import (LabelMalignancy, LabelSpiculation, LabelSubtlety,
                         LabelTexture, LabelVolume)
from .label_lndb import LabelLNDbTexture, LabelLNDbVolume
from .label_stanfordradiogenomics import LabelStfRG, LabelStfRGGender, LabelStfRGSmoking

LABEL_DICT = {'volume': LabelVolume,
              'malignancy': LabelMalignancy,
              'texture': LabelTexture,
              'spiculation': LabelSpiculation,
              'subtlety': LabelSubtlety,

              'LNDbTaskVolume': LabelLNDbVolume,
              'LNDbTaskTexture': LabelLNDbTexture,

              "StfRG": LabelStfRG,
              "StfRGGender": LabelStfRGGender,
              "StfRGSmoking": LabelStfRGSmoking
              }
