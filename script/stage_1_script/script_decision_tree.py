from src.stage_1_code.Dataset_Loader import Dataset_Loader
from src.stage_1_code.Method_DT import Method_DT
from src.stage_1_code.Result_Saver import Result_Saver
from src.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
from src.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from src.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np


#---- Decision Tree script ----
if 1:
    #---- parameter section -------------------------------
    None
    np.random.seed(1)
    #------------------------------------------------------
    
    #---- objection initialization section ---------------
    data_obj = Dataset_Loader('toy', '')
    data_obj.dataset_source_folder_path = '../../data/stage_1_data/'
    data_obj.dataset_source_file_name = 'toy_data_file.txt'
    
    method_obj = Method_DT('decision tree', '')
    
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_1_result/DT_'
    result_obj.result_destination_file_name = 'prediction_result'
    
    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Train_Test_Split('train test split', '')
    
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    #------------------------------------------------------
    
    #---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('Decision Tree Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    #------------------------------------------------------
    
    
    
    