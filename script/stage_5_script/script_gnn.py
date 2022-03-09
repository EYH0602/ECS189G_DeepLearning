from src.stage_5_code.Dataset_Loader_Node_Classification import DatasetLoader
from src.stage_5_code.Method_GNN import MethodGNN
from src.stage_5_code.Result_Saver import ResultSaver
from src.stage_5_code.Setting_Train_Test import SettingTrainTest
from src.stage_5_code.Evaluate_Accuracy import EvaluateAccuracy
from src.stage_5_code.Training_Conv_Plotter import Plotter
import numpy as np
import torch

#---- Graph Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage 5', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/'
    data_obj.dataset_name = 'cora'

    method_obj = MethodGNN('graph neural network', '')
    
    if torch.cuda.is_available():
        print("training on: cuda")
        method_obj = method_obj.cuda()

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingTrainTest('train test', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    
    plotter = Plotter()
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.plotter = Plotter
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('GNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    
    plotter.plot("stage_5_plot.png")
    