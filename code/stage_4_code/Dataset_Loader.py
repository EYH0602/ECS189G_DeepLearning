from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

class DatasetLoader(dataset):
    date = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        train = self.load_individual(folder_path=self.dataset_source_folder_path, train=True, pos=True)
        train += self.load_individual(folder_path=self.dataset_source_folder_path, train=True, pos=False)

        test = self.load_individual(folder_path=self.dataset_source_folder_path, train=False, pos=True)
        test += self.load_individual(folder_path=self.dataset_source_folder_path, train=False, pos=False)

        return {'train': train, 'test': test}



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
            data += [{'words': self.tokenize(open(source_path + file, "r", encoding='utf-8').read()), 'label': label}]
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