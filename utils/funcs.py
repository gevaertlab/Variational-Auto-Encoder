''' Util functions '''

import os
import yaml

def getVersion(path):
    ''' Get current version by increment the previous version '''
    folder_lst = os.listdir(path)
    return max([int(folder.split('_')[1]) for folder in folder_lst]) + 1
    
    
def saveConfig(path, config_file):
    with open(os.path.join(path, 
                    'config.yaml'), 
            'w') as f:
        yaml.dump(config_file, f)