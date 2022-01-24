'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path +
                 self.dataset_source_file_name, 'r')
        reader = csv.reader(f)
        for line in reader:
            X.append(line[1:])  # remaining 784 are features
            y.append(line[0])   # first element is label
        f.close()
        return {'X': X, 'y': y}
