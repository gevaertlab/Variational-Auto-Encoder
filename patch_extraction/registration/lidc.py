
import pylidc as pl
from tqdm import tqdm
from multiprocessing import Manager, Pool, Process
import os
from .register import Register


class LidcReg(Register):

    def __init__(self,
                 name='LIDC',
                 root_path='/labs/gevaertlab/data/lung cancer/TCIA_LIDC',
                 info_dict_save_path='/labs/gevaertlab/data/lung cancer/TCIA_LIDC/info_dict.json'):
        super().__init__(name, root_path, info_dict_save_path=info_dict_save_path)

    def register(self, multi=False):
        if os.path.exists(self.info_dict_save_path):
            self.info_dict = self.load_json(self.info_dict_save_path)
        scan_list = list(pl.query(pl.Scan))
        if not multi:
            for scan in tqdm(scan_list):
                self.get_lidc_info(scan)
        else:
            # HACK: cannot be achieved.
            pool = Pool(processes=os.cpu_count())
            data_list = pool.starmap(self.get_lidc_info,
                                     [(s, True) for s in scan_list])
            data_dict = {data[0]: (data[1], data[2]) for data in data_list}
            self.info_dict['data_dict'].update(data_dict)
        self.save_info_dict()
        pass

    def get_lidc_info(self, scan, multi=False):
        pid = scan.patient_id
        nodules = scan.annotations
        for nodule in nodules:
            file_name = f"{pid}.{nodule.id}"
            y, x, z = nodule.centroid
            center_point = (x, y, z)
            img_path = scan.get_path_to_dicom_files()
            if multi:
                return (file_name, img_path, center_point)
            else:
                self.info_dict['data_dict'][file_name] = (
                    img_path, center_point)
                return None