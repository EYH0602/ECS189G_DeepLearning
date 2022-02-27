"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_4_code.Dictionary import Dictionary
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader


class SettingTrainTest(setting):
    plotter = None

    def load_run_save_evaluate(self, batch_size):
        loaded_data = DataLoader(self.dataset, batch_size=batch_size)

        # run MethodModule
        self.method.data = loaded_data
        self.method.plotter = self.plotter
        self.method.dataset = self.dataset
        
        self.method.run()
