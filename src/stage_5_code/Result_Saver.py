"""
Concrete ResultModule class for a specific experiment ResultModule output
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.result import result
import pickle


class ResultSaver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        result_file = self.result_destination_folder_path + \
                      self.result_destination_file_name + \
                      '_' + str(self.fold_count)
        f = open(result_file, 'wb')
        pickle.dump(self.data, f)
        f.close()
