"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_4_code.Dictionary import Dictionary
from sklearn.model_selection import train_test_split
import numpy as np


class SettingTrainTest(setting):
    plotter = None

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        # self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.data = loaded_data
        self.method.plotter = self.plotter
        
        dic = Dictionary()
        for sec in loaded_data['train']['X']:
            dic.update_frequencies_by_sentence(sec)
        dic.compute_dictionary(1000)
        self.method.word_dict = dic
        
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate_accuracy(), None
