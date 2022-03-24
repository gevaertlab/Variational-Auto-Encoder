from .label_lidc import (LabelMalignancy, LabelSpiculation, LabelSubtlety,
                         LabelTexture, LabelVolume)
from .label_lndb import LabelLNDbTexture, LabelLNDbVolume
from .label_stanfordradiogenomics import (LabelStfAJCC, LabelStfNStage,
                                          LabelStfReGroup, LabelStfRG,
                                          LabelStfRGGender, LabelStfRGSmoking,
                                          LabelStfTStage, LabelStfHisGrade,
                                          LabelStfRGPleuralInvasion, LabelStfRGVolume,
                                          LabelStfRGLymphInvasion, LabelStanfordRadiogenomics,
                                          LabelStfEGFRMutation, LabelStfKRASMutation)

LABEL_DICT = {'volume': LabelVolume,
              'malignancy': LabelMalignancy,
              'texture': LabelTexture,
              'spiculation': LabelSpiculation,
              'subtlety': LabelSubtlety,

              'LNDbTaskVolume': LabelLNDbVolume,
              'LNDbTaskTexture': LabelLNDbTexture,

              "StfRadiogenomics": LabelStanfordRadiogenomics,  # meta label
              "StfRG": LabelStfRG,  # meta label
              "StfVolume": LabelStfRGVolume,
              "StfRGGender": LabelStfRGGender,  # only test
              "StfRGSmoking": LabelStfRGSmoking,  # only test
              "StfLymphInvasion": LabelStfRGLymphInvasion,
              "StfPleuralInvasion": LabelStfRGPleuralInvasion,

              "StfReGroup": LabelStfReGroup,  # meta label
              "StfTStage": LabelStfTStage,
              "StfNStage": LabelStfNStage,
              "StfAJCC": LabelStfAJCC,
              "StfHisGrade": LabelStfHisGrade,
              "StfEGFRMutation": LabelStfEGFRMutation,
              "StfKRASMutation": LabelStfKRASMutation,
              }
