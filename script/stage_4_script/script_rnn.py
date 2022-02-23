from code.stage_4_code.Dataset_Loader import DatasetLoader
from code.stage_4_code.Method_RNN import MethodRNN
from code.stage_4_code.Result_Saver import ResultSaver
from code.stage_4_code.Setting_Train_Test import SettingTrainTest
from code.stage_4_code.Evaluate_Accuracy import EvaluateAccuracy
from code.stage_4_code.Training_Conv_Plotter import Plotter
import numpy as np
import torch

#---- RNN script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage 4', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    method_obj = MethodRNN('RNN', '')

    if torch.cuda.is_available():
        print("training on: cuda")
        method_obj = method_obj.cuda()

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingTrainTest('train test', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    
    plotter = Plotter()
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.plotter = plotter
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    
    plotter.plot("stage_4_plot.png")
    