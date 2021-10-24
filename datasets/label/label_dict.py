from .label_lidc import (LabelMalignancy,
                         LabelSpiculation,
                         LabelSubtlety,
                         LabelTexture,
                         LabelVolume)


LABEL_DICT = {'volume': LabelVolume,
              'malignancy': LabelMalignancy,
              'texture': LabelTexture,
              'spiculation': LabelSpiculation,
              'subtlety': LabelSubtlety}
