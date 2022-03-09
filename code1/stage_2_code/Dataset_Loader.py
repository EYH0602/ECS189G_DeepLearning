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
    test_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load_single(self, file_name):
        print('loading data from {}...'.format(file_name))
        X = []
        y = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                elements = [int(i) for i in line]
                X.append(elements[1:])
                y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

    def load(self):
        return {'train_data': self.load_single(self.dataset_source_file_name),
                'test_data': self.load_single(self.test_source_file_name)}