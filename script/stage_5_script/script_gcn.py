from code1.stage_5_code.Method_GCN import Method_GCN
from code1.stage_5_code.Result_Saver import Result_Saver
from code1.stage_5_code.Setting_Train_Test import Setting_Train_Test
from code1.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code1.stage_5_code.Training_Conv_Plotter import Plotter
from code1.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed'
    data_obj.dataset_name = 'pubmed'

    method_obj = Method_GCN('Graph Convolutional Networks', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train, test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.plotter = Plotter()
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------


