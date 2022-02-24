from code.stage_3_code.Dataset_Loader import DatasetLoader
from code.stage_3_code.Method_CNN import MethodCNN
from code.stage_3_code.Result_Saver import ResultSaver
from code.stage_3_code.Setting_Train_Test import SettingTrainTest
from code.stage_3_code.Evaluate_Accuracy import EvaluateAccuracy
from code.stage_3_code.Training_Conv_Plotter import Plotter
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage 3', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'MNIST'

    method_obj = MethodCNN('Convolutional Neural Network', '')

    result_obj = ResultSaver('saver', 'MNIST')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
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
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    
    plotter.plot("stage_3_plot.png")
    