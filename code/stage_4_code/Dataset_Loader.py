"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import os
from code.base_class.dataset import dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


class DatasetLoader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None, n_file=400,):
        super().__init__(dName, dDescription)
        self.n_file = n_file

    def load(self):
        train_pos = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=True, pos=True)
        train_neg = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=True, pos=False)
        
        train = {
            'X': train_pos['X'] + train_neg['X'],
            'y': train_pos['y'] + train_neg['y']
        }

        test_pos = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=False, pos=True)
        test_neg = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=False, pos=False)
        
        test = {
            'X': test_pos['X'] + test_neg['X'],
            'y': test_pos['y'] + test_neg['y']
        }

        return {'train': train, 'test': test}

    def load_individual(self, folder_path: str, train: bool, pos: bool):
        data = {'X': [], 'y':[]}
        dataset_name = ''
        dataset_label = ''
        label = 0
        if train:
            dataset_name = 'train/'
        else:
            dataset_name = 'test/'
        if pos:
            dataset_label = 'pos/'
            label = 1
        else:
            dataset_label = 'neg/'
        source_path = folder_path + dataset_name + dataset_label

        i = 0 
        for file in os.listdir(source_path):
            if i >= self.n_file:
                break
            data['X'].append(self.tokenize(open(source_path + file, "r", encoding='utf-8').read()))
            data['y'].append(label)
            i += 1
        

        return data

    def tokenize(self, s: str):
        # source: https://machinelearningmastery.com/clean-text-machine-learning-python/
        tokens = word_tokenize(s)
        tokens = [token.lower() for token in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        return [word for word in words if (word not in stop_words)]
