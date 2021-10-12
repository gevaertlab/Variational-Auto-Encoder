

AUG_PARAMS = {'rotate':
              (  # each process is a tuple of operation + repeat
                  [  # operations is  list
                      ('rotate',  # function
                       {'range': (-45, 45)})  # param: rotation angle ranges
                  ],
                  5
              ),  # num of augmented images for each lesion,
              'shift':
              (
                  [
                      ('shift',
                       {'range': (-12, 12)})
                  ],
                  5
              )
              }
