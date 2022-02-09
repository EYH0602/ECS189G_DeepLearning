"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import pickle
from code.base_class.dataset import dataset


class DatasetLoader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def load(self):
        print('loading data...')
        f = open(
            self.dataset_source_folder_path +\
            self.dataset_source_file_name,
            "rb"
        )
        data = pickle.load(f)
        f.close()
        
        return data
