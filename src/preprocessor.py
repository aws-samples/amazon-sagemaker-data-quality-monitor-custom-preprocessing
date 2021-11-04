import json
import random
import ast
import sys
import subprocess
import os

METADATA_CUSTOM_ATTR_TEST_INDICATOR = 'testIndicator'
LABEL = 'Churn'

def write_to_file(log, filename):
    with open(f"/opt/ml/processing/output/{filename}.log", "a") as f:
        f.write(log + '\n')
        
def str_to_bool(s):
    return s.lower() in ("true", "t", "1", 'y')

def test_indicator_exist(attribute):
    return (True if METADATA_CUSTOM_ATTR_TEST_INDICATOR in attribute else False)
    
def eval_test_indicator(attribute):
    if test_indicator_exist(attribute):
        return str_to_bool(attribute[METADATA_CUSTOM_ATTR_TEST_INDICATOR])
    else:
        return False

def get_class_val(probability):
    v = ast.literal_eval(probability)
    return (int(v[0] >= 0.5) if isinstance(v, list) else int(v >= 0.5))

def preprocess_handler(inference_record):
    #*********************
    # a single inference implementation
    #*********************
    input_enc_type = inference_record.endpoint_input.encoding
    input_data = inference_record.endpoint_input.data.rstrip("\n")
    output_data = get_class_val(inference_record.endpoint_output.data.rstrip("\n"))
    eventmedatadata = inference_record.event_metadata
    custom_attribute = json.loads(eventmedatadata.custom_attribute[0]) if eventmedatadata.custom_attribute is not None else None
    is_test = eval_test_indicator(custom_attribute) if custom_attribute is not None else True
    
    if is_test:
        return []
    elif input_enc_type == "CSV":
        outputs = output_data+','+input_data
        return {str(i).zfill(20) : d for i, d in enumerate(outputs.split(","))}
    elif input_enc_type == "JSON":  
        outputs = {**{LABEL: output_data}, **json.loads(input_data)}
        write_to_file(str(outputs), "log")
        return {str(i).zfill(20) : outputs[d] for i, d in enumerate(outputs)}
    else:
        raise ValueError(f"encoding type {input_enc_type} is not supported") 
        
        


