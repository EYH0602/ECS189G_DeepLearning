"""
Concrete IO class for a specific dataset
"""

import os
import json
from code.base_class.dataset import dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch


class DatasetLoader(dataset):
    data = None
    dataset_source_folder_path = None
    train_datafile_path = None
    test_datafile_path = None
    max_vocab_size = 250000
    batch_size = 64
    
    TEXT = None
    
    def load(self):
        train_path = self.dataset_source_folder_path + self.train_datafile_path
        test_path = self.dataset_source_folder_path + self.test_datafile_path
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            self.prepare_data()

        # load datasets
        TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)
        LABEL = data.LabelField(dtype = torch.float)
        self.TEXT = TEXT
        fields = {
            'text': ('text', TEXT),
            'label': ('label', LABEL)
        }
        train_data = data.TabularDataset(train_path, 'json', fields)
        test_data = data.TabularDataset(test_path, 'json', fields)

        # build vocab
        TEXT.build_vocab(train_data,
                         max_size=self.max_vocab_size,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        # split batches
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=self.batch_size,
            device=device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True
        )

        return {'train': train_iterator, 'test': test_iterator}

    def load_source_data(self):
        train_pos = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=True, pos=True)
        train_neg = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=True, pos=False)

        train = train_pos + train_neg
        train_json_str = ""
        for train_instance in train:
            train_json_str += json.dumps(train_instance)

        test_pos = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=False, pos=True)
        test_neg = self.load_individual(
            folder_path=self.dataset_source_folder_path, train=False, pos=False)

        test = test_pos + test_neg
        test_json_str = ""
        for test_instance in test:
            test_json_str += json.dumps(test_instance)

        with open(self.train_datafile_path, 'w') as f:
            f.write(train_json_str)
        with open(self.test_datafile_path, 'w') as f:
            f.write(test_json_str)

    def prepare_data(self):
        self.load_source_data()

    def load_individual(self, folder_path: str, train: bool, pos: bool):
        data = []
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

        for file in os.listdir(source_path):
            data.append({
                'text': self.tokenize(open(source_path + file, "r", encoding='utf-8').read()),
                'label': label
            })

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
