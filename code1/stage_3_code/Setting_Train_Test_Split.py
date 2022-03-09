'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from matplotlib import pyplot as plt

from code.base_class.setting import setting
import numpy as np

class Setting_Train_Test_Split(setting):
    plotter = None
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()
        train_data = loaded_data['train']
        test_data = loaded_data['test']

        # run MethodModule
        # X_train = [pair['image'] for pair in train_data]
        # y_train = [pair['label'] for pair in train_data]

        X_test, y_test = [pair['image'] for pair in test_data], [pair['label'] for pair in test_data]
        self.method.data = {'train': train_data, 'test': {'X': X_test, 'y': y_test}}
        self.method.plotter = self.plotter
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate_accuracy(), None

