'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        data = self.dataset.load()
        train_data = data['train_data']
        test_data = data['test_data']

        score_list = []

        X_train, y_train = np.array(train_data['X']), np.array(train_data['y'])
        X_test, y_test = np.array(test_data['X']), np.array(test_data['y'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result

        self.result.save()
            
        self.evaluate.data = learned_result
        score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        