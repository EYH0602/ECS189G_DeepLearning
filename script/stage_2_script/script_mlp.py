from code.stage_2_code.Dataset_Loader import DatasetLoader
from code.stage_2_code.Method_MLP import MethodMLP
from code.stage_2_code.Result_Saver import ResultSaver
from code.stage_2_code.Setting_Train_Test import SettingTrainTest
from code.stage_2_code.Evaluate_Accuracy import EvaluateAccuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage 2', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.train_dataset_source_file_name = 'train.csv'
    data_obj.test_dataset_source_file_name = 'test.csv'

    method_obj = MethodMLP('multi-layer perceptron', '')

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingTrainTest('train test', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    