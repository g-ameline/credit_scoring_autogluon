import datetime
import os
import numpy


data_url = 'https://assets.01-edu.org/ai-branch/project5/home-credit-default-risk.zip'

root_folder_path='..'
data_folder_name = 'data'
data_folder_path = os.path.join(root_folder_path,data_folder_name)
results_folder_name = 'results'
results_folder_path = os.path.join(root_folder_path,results_folder_name)
scripts_folder_name = 'scripts'
scripts_folder_path = os.path.join(root_folder_path,scripts_folder_name)

client_output_folder_name = 'clients_outputs'
client_output_folder_path = os.path.join(results_folder_path,client_output_folder_name)

submission_file_name = 'submission.csv'
submission_file_path = os.path.join(results_folder_path, submission_file_name)

submission_result_file_name = 'submission_result.png'
submission_result_file_path = os.path.join(results_folder_path, submission_result_file_name)

model_folder_name = 'model'
model_folder_path = os.path.join(results_folder_path,model_folder_name)

model_with_learning_curves_folder_name = 'model_with_learning_curves'
model_with_learning_curves_folder_path = os.path.join(results_folder_path,model_with_learning_curves_folder_name)

dashboard_folder_name = 'dashboard'
dashboard_folder_path = os.path.join(results_folder_path,dashboard_folder_name)

feature_importances_tab_file_name = 'feature_importances.csv'
feature_importances_tab_file_path = os.path.join(dashboard_folder_path,feature_importances_tab_file_name)

feature_importances_hist_file_name = 'feature_importances.png'
feature_importances_hist_file_path = os.path.join(dashboard_folder_path,feature_importances_hist_file_name)

models_graph_file_name = 'models_graph.png'
models_graph_file_path = os.path.join(model_folder_path, models_graph_file_name)

downloaded_data_file_name = 'home-credit-default-risk.zip'
downloaded_data_file_path= os.path.join(data_folder_path, downloaded_data_file_name)
unzipped_data_folder_name = 'unzipped_data'
unzipped_data_folder_path = os.path.join(data_folder_path, unzipped_data_folder_name)

unzipped_files = {
    # info/description
    'HomeCredit_columns_description.csv',
    # input data
    'POS_CASH_balance.csv',
    'bureau.csv',
    'bureau_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'previous_application.csv',
    # 
    'application_test.csv',
    'application_train.csv',
    'sample_submission.csv',
}

needed_unzipepd_files = {
    'application_train.csv',
    'application_test.csv',
}

unzipped_train_data_file_name = 'application_train.csv'
unzipped_train_data_file_path = os.path.join(unzipped_data_folder_path, unzipped_train_data_file_name)
unzipped_test_data_file_name = 'application_test.csv'
unzipped_test_data_file_path = os.path.join(unzipped_data_folder_path, unzipped_test_data_file_name)

train_data_file_name = 'train.csv'
train_data_file_path = os.path.join(data_folder_path, train_data_file_name)
test_data_file_name = 'test.csv'
test_data_file_path = os.path.join(data_folder_path, test_data_file_name)
    

# project
# │   README.md
# │
# └───data
# │   │   ...
# │
# └───results
# │   │
# │   └───model (free format)
# │   │   │   my_own_model.pkl
# │   │   └   model_report.txt
# │   │
# │   └feature_engineering
# │   │   └   EDA.ipynb
# │   │
# │   └───clients_outputs
# │   │   │   client1_correct_train.pdf  (free format)
# │   │   │   client2_wrong_train.pdf  (free format)
# │   │   └   client_test.pdf   (free format)
# │   │
# │   └───dashboard (optional)
# │   │   │   dashboard.py  (free format)
# │   │   │   ...
# │
# └───scripts (free format)
# │   │   train.py
# │   │   predict.py
# │   │   preprocess.py

