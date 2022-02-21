from code.stage_4_code.Dataset_Loader import DatasetLoader
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage 4', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/stage_4_data/text_classification/'

