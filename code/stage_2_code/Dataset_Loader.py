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
    train_dataset_source_file_name = None
    test_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_csv(self, fileName):
        f = open(self.dataset_source_folder_path + fileName, 'r')
        X = []
        y = []
        reader = csv.reader(f)

        for line in reader:
            X.append(list(map(int, line[1:]))) # 1 to 784 are all data
            y.append(int(line[0])) # 0 is label
        f.close()


    def load(self):
        print('loading data...')
        train_data = self.load_csv(self.train_dataset_source_file_name)
        test_data = self.load_csv(self.test_dataset_source_file_name)

        return {'train': train_data, 'test': test_data}