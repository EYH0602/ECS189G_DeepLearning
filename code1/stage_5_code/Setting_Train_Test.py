'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from matplotlib import pyplot as plt

from code1.base_class.setting import setting
import numpy as np

class Setting_Train_Test(setting):
    plotter = None
    def load_run_save_evaluate(self):
        # load dataset
        self.method.data = self.dataset.load()
        self.method.plotter = self.plotter
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate_accuracy(), None

