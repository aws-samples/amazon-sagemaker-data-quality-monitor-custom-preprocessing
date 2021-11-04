import json
import os
import pickle as pkl
import json
import numpy as np
import pandas as pd
import sagemaker_xgboost_container.encoder as xgb_encoders
import xgboost as xgb
from io import StringIO
from io import BytesIO

#For Gunicorn/Flask xgboost image, we need to ensure input and output encoding match exactly for model monitor (CSV or JSON)
from flask import Response 

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster

def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """
    if request_content_type == "text/libsvm":
        return xgb_encoders.libsvm_to_dmatrix(request_body)
    
    elif request_content_type == "text/csv":
        return xgb_encoders.csv_to_dmatrix(request_body.rstrip("\n"))
    
    elif request_content_type == "application/json":
        #-----------------
        # single input implementation
        #-----------------
        request = json.loads(request_body)
        feature = ",".join(request.values())
        return xgb_encoders.csv_to_dmatrix(feature.rstrip("\n"))
    else:
        raise ValueError("Content type {} is not supported.".format(request_content_type))
        
def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.
    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    is_feature_importance = False
    prediction = model.predict(input_data)
    if is_feature_importance:
        feature_contribs = model.predict(input_data, 
                                         pred_contribs=True, 
                                         validate_features=False)
        output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    else:
        output = prediction
    return output


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    print(predictions)
    if content_type == "text/csv":
        #result = ",".join(str(x) for x in predictions[0])
        result = ",".join(str(x) for x in predictions)
        return Response(result, mimetype=content_type)
    elif content_type == "application/json":
        result = json.dumps(predictions.tolist())
        return Response(result, mimetype=content_type)
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
