import os, shutil
import sys
import time 
import pandas as pd, numpy as np
import pprint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import linregress

sys.path.insert(0, './../app')
import algorithm.utils as utils  
import algorithm.model_trainer as model_trainer
import algorithm.model_server as model_server
import algorithm.model_tuner as model_tuner
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.recommender as recommender

inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")
data_path = os.path.join(inputs_path, "data")

train_data_path = os.path.join(data_path, "training", "recommenderBaseMainInput")
test_data_path = os.path.join(data_path, "testing", "recommenderBaseMainInput")

model_path = "./ml_vol/model/"
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_results")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)


# change this to whereever you placed your local testing datasets
local_datapath = "./../../datasets" 


'''
this script is useful for doing the algorithm testing locally without needing
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing.
This isnt foolproof. You can still have host os or python version-related issues, so beware.
'''

model_name= recommender.MODEL_NAME


def create_ml_vol():
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": {
                        "recommenderBaseMainInput": None
                    },
                    "testing": {
                        "recommenderBaseMainInput": None
                    }
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            },
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,
            }
        }
    }
    def create_dir(curr_path, dir_dict):
        for k in dir_dict:
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None:
                create_dir(dir_path, dir_dict[k])
    create_dir("", dir_tree)


def copy_example_files(dataset_name):
    # data schema
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # train data    
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_train.csv", os.path.join(train_data_path, f"{dataset_name}_train.csv"))    
    # test data     
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_test.csv", os.path.join(test_data_path, f"{dataset_name}_test.csv"))    
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json"))


def run_HPT(num_hpt_trials):
    # Read data
    train_data = utils.get_data(train_data_path)
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    model_tuner.tune_hyperparameters(train_data, data_schema, num_hpt_trials, hyper_param_path, hpt_results_path)


def train_and_save_algo():
    # Read hyperparameters
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)
    # Read data
    train_data = utils.get_data(train_data_path)
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)
    # get trained preprocessor, model, training history
    preprocessor, model = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)
    # Save the processing pipeline
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model
    recommender.save_model(model, model_artifacts_path)
    print("done with training")


def load_and_test_algo():
    # Read data
    test_data = utils.get_data(test_data_path)
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)
    # instantiate the trained model
    predictor = model_server.ModelServer(model_artifacts_path)
    # make predictions
    predictions = predictor.predict(test_data, data_schema)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    results = score(test_data, predictions)  
    print("done with predictions")
    return results


def set_id_and_target_cols(dataset_name):
    global id_col, target_col, test_key
    data_schema = utils.get_data_schema(data_schema_path)
    # set the id attribute
    id_col = data_schema["inputDatasets"]["recommenderBaseMainInput"]["idField"]       
    # set the target attribute
    target_col = data_schema["inputDatasets"]["recommenderBaseMainInput"]["targetField"]   
    # test_key
    test_key = pd.read_csv(f"{local_datapath}/{dataset_name}/{dataset_name}_test_key.csv")


def score(test_data, predictions):
    predictions = predictions.merge(test_key[[id_col, target_col]], on=id_col)
    rmse = mean_squared_error(predictions[target_col], predictions['prediction'], squared=False)
    mae = mean_absolute_error(predictions[target_col], predictions['prediction'])
    # r2 = r2_score(predictions[target_col], predictions['prediction'])
    _, _, r_value, _, _  = linregress(predictions[target_col], predictions['prediction'])
    r2 = r_value * r_value
    
    print("act mean", predictions[target_col].mean())
    print("pred mean", predictions['prediction'].mean())
    
    q3, q1 = np.percentile(predictions[target_col], [75, 25])
    iqr = q3 - q1
    nmae = mae / iqr
    scores = {
        "rmse": np.round(rmse,4), 
        "mae": np.round(mae,4),
        "nmae": np.round(nmae,4),
        "r2": np.round(r2,4),
        "perc_pred_missing": np.round( 100 * (1 - predictions.shape[0] / test_data.shape[0]), 2)
        }
    return scores


def save_test_outputs(results, run_hpt, dataset_name):    
    df = pd.DataFrame(results) if dataset_name is None else pd.DataFrame([results])        
    df = df[["model", "dataset_name", "run_hpt", "num_hpt_trials", 
             "rmse", "mae", "nmae", "r2", "perc_pred_missing",
             "elapsed_time_in_minutes"]]    
    print(df)
    file_path_and_name = get_file_path_and_name(run_hpt, dataset_name)
    df.to_csv(file_path_and_name, index=False)


def get_file_path_and_name(run_hpt, dataset_name): 
    if dataset_name is None: 
        fname = f"_{model_name}_results_with_hpt.csv" if run_hpt else f"_{model_name}_results_no_hpt.csv"
    else: 
        fname = f"{model_name}_{dataset_name}_results_with_hpt.csv" if run_hpt else f"{model_name}_{dataset_name}_results_no_hpt.csv"
    full_path = os.path.join(test_results_path, fname)
    return full_path


def run_train_and_test(dataset_name, run_hpt, num_hpt_trials):
    start = time.time() 
    
    create_ml_vol()   # create the directory which imitates the bind mount on container
    copy_example_files(dataset_name)   # copy the required files for model training    
    if run_hpt: run_HPT(num_hpt_trials)               # run HPT and save tuned hyperparameters
    train_and_save_algo()        # train the model and save
    
    set_id_and_target_cols(dataset_name=dataset_name)
    results = load_and_test_algo()        # load the trained model and get predictions on test data
    
    end = time.time()
    elapsed_time_in_minutes = np.round((end - start)/60.0, 2)
    
    results = { **results,  
               "model": model_name, 
               "dataset_name": dataset_name, 
               "run_hpt": run_hpt, 
               "num_hpt_trials": num_hpt_trials if run_hpt else None, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes 
               }
    
    return results


if __name__ == "__main__":
    
    num_hpt_trials = 5
    run_hpt_list = [False, True]
    run_hpt_list = [False]
    
    datasets = ["amazon_electronics_small", "anime", "jester", "modcloth", "book_crossing_small", "movielens_1m", "movielens_10m"]
    datasets = ["movielens_1m"]
    
    for run_hpt in run_hpt_list:
        all_results = []
        for dataset_name in datasets:        
            print("-"*60)
            print(f"Running dataset {dataset_name}")
            results = run_train_and_test(dataset_name, run_hpt, num_hpt_trials)
            save_test_outputs(results, run_hpt, dataset_name)            
            all_results.append(results)
            print("-"*60)
        
        save_test_outputs(all_results, run_hpt, dataset_name=None)