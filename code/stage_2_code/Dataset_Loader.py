"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
from code.base_class.dataset import dataset


class DatasetLoader(dataset):
    data = None
    dataset_source_folder_path = None
    test_dataset_source_file_name = None
    train_dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def __load_csv(self, filename):
        f = open(self.dataset_source_folder_path + filename, 'r')
        X = []
        y = []
        reader = csv.reader(f)

        for line in reader:
            X.append(list(map(int, line[1:])))  # remaining 784 are features
            y.append(int(line[0]))   # first element is label
        f.close()

        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')
        train_data = self.__load_csv(self.train_dataset_source_file_name)
        test_data = self.__load_csv(self.test_dataset_source_file_name)
        return {'train': train_data, 'test': test_data}
