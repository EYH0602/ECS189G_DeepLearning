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
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        data = []
        file = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        file.readline()
        for line in file.readlines():
            data.append(self.tokenize(line[line.index(',') + 2:len(line) - 2]))
        return data


    def tokenize(self, s: str):
        # source: https://machinelearningmastery.com/clean-text-machine-learning-python/
        tokens = word_tokenize(s)
        tokens = [token.lower() for token in tokens]
        return tokens
