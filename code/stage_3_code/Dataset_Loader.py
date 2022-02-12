'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import pickle
import matplotlib as plt
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        if 1:
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
            data = pickle.load(f)
            f.close()
            # get data visuals
            # print('training set size:', len(data['train']), 'testing set size: ', len(data['test']))
        #for pair in data['train']:
        # for pair in data['test']
            #plt.imshow(pair['image'], cmap = "Greys")
            #plt.show()
            #print(pair['label'])