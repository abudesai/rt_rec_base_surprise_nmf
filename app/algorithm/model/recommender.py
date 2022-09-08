
import json
import sys

import numpy as np, pandas as pd
import os
import joblib


import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.matrix_factorization import NMF

MODEL_NAME = "recommender_base_nmf_surprise"


model_fname = "model.save"
model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


class Recommender:
    def __init__(self):
        self.ma = None
        self.mi = None
        self.model = NMF()
        return

    def fit(self, train_X, train_y):
        self.ma = int(max(train_y) + 1)
        self.mi = int(min(train_y) - 1)
        reader = Reader(rating_scale=(self.mi, self.ma))

        train = pd.DataFrame(train_X.copy())
        train["rating"] = train_y
        data = Dataset.load_from_df(train, reader)
        self.model.fit(data.build_full_trainset())
        return

    def predict(self, X):
        test = pd.DataFrame(X)
        test["rating"] = 1
        test_ = Dataset.load_from_df(test, reader=Reader(rating_scale=(self.mi, self.ma))).build_full_trainset()
        testset = test_.build_testset()
        predictions = pd.DataFrame(self.model.test(testset))
        temp2 = pd.merge(test, predictions, left_on=[0, 1], right_on=['uid', 'iid'], how="left")
        return np.array(temp2.est).reshape(-1, 1)

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        y_pred = (self.predict(x_test)).reshape(-1)
        y_test = np.array(y_test).reshape(-1)
        return np.mean((y_pred - np.array(y_test)) ** 2)

    def save(self, model_path):        

        model_params = {}
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        
        self.model.pu = self.model.pu.base
        self.model.qi = self.model.qi.base
        self.model.bu = self.model.bu.base
        self.model.bi = self.model.bi.base
        
        # surprise.dump.dump(os.path.join(model_path, model_fname), predictions=None, algo=self.model, verbose=1)
        surprise.dump.dump(os.path.join(model_path, model_fname), algo=self.model)
        np.save(os.path.join(model_path, "lower_bound.npy"), self.mi)
        np.save(os.path.join(model_path, "upper_bound.npy"), self.ma)

    @classmethod
    def load(ml, model_path):
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        mf = ml(**model_params)
        mf.mi = int(np.load(os.path.join(model_path, "lower_bound.npy")))
        mf.ma = int(np.load(os.path.join(model_path, "upper_bound.npy")))
        mf.model = surprise.dump.load(os.path.join(model_path, model_fname))[1]
        return mf


def get_data_based_model_params(X):
    """
    returns a dictionary with N: number of users and M = number of items
    This assumes that the given numpy array (X) has users by id in first column,
    and items by id in 2nd column.
    The ids must be contiguous i.e. 0 to N-1 and 0 to M-1 for users and items.
    """
    return {}


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = Recommender.load(model_path)
    except:
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path):
    with open(os.path.join(f_path, history_fname), mode='w') as f:
        f.write(json.dumps(history, indent=2))
