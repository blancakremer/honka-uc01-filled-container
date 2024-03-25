"""
This module defines the `model_training` function used by the pipeline orchestrator to train a machine 
learning model using ElasticNet regularization. This function defines the logic for training the model and evaluating 
its performance.

Any additional functions or utilities required for this step can be defined within this script itself or split into 
different scripts and included in the Process directory.
"""

from typing import Dict, Any
from sklearn.linear_model import ElasticNet
from datetime import datetime
import sklearn as skl
from Models import utils
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#import tensorflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
import numpy as np
import config
from mlflow.tracking.client import MlflowClient
import time
print(">current version of sklearn: ", skl.__version__)
def model_training(data: Dict[str, Any]):
    """
    Args:
        data: A dictionary containing the preprocessed data.

    Returns:
        None
    """

    # ADD YOUR CODE HERE: READ INPUT DATA
    # ADD YOUR CODE HERE: TRAIN THE MODEL
    # ADD YOUR CODE HERE: DO NOT FORGET TO TRACK THE MODEL TRAINING
    inp_len = len(data["dataset1_x_train"][0])
    #inp_len_y = len(data["dataset1_y_train"][0])
    #x_train = np.array(data["dataset1_x_train"])
    #x_train = x_train.reshape(-1, 1, inp_len)
    #y_train = np.array(data["dataset1_y_train"])
    #y_train = y_train.reshape(-1, 1, 1)

    #x_test = np.array(data["dataset1_x_test"])
    #x_test = x_test.reshape(-1, 1, inp_len)
    #y_test = np.array(data["dataset1_y_test"])
    #y_test = y_test.reshape(-1, 1, 1)
    # Hidden layer with a lot more feature extraction neurons
    #model.add(tensorflow.keras.layers.Conv1D(filters=32,kernel_size=2,activation="swish",input_shape=(1,6)))
    #model.add(tensorflow.keras.layers.Dense(54, activation=tensorflow.keras.activations.swish))
    #model.add(tensorflow.keras.layers.Dense(54, activation='swish'))
    #model.add(tensorflow.keras.layers.Dense(54, activation='swish'))
    #model.add(tensorflow.keras.layers.LSTM(100,input_shape=(1,inp_len), return_sequences=True,activation="swish"))
    #model.add(tensorflow.keras.layers.Dropout(0.1))
    #model.add(tensorflow.keras.layers.LSTM(100, return_sequences=True, activation="swish"))
    #model.add(tensorflow.keras.layers.Dropout(0.1))
    #model.add(tensorflow.keras.layers.LSTM(60, return_sequences=True, activation="swish"))
    #model.add(tensorflow.keras.layers.Dropout(0.6))
    x_train = data["dataset1_x_train"]
    y_train = data["dataset1_y_train"]
    x_test = data["dataset1_x_test"]
    y_test = data["dataset1_y_test"]
    client = MlflowClient(config.MLFLOW_ENDPOINT)
    mlflow.set_tracking_uri(config.MLFLOW_ENDPOINT)
    try:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    except:
        time.sleep(10)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="RFRModel"):
        regr = RandomForestRegressor(max_depth=7, random_state=0,n_estimators=200)
        regr.fit(x_train, y_train)

        pred_values = regr.predict(x_test)

        (rmse, mae, r2) = utils.eval_metrics_2(y_test, pred_values)

        print(f"RRF model (depth={7:f}, n_estimator={200:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("depth", 7)
        mlflow.log_param("n_estimator", 200)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = regr.predict(x_train)
        signature = infer_signature(np.array(x_train), np.array(predictions))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                regr, "model", registered_model_name="RFRModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(regr, "model", signature=signature)

    pass
