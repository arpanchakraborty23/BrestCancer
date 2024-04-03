import os,sys
import pickle
import pandas as pd 
import numpy as np 


from src.exception import CustomException
from src.logger import logging

def load_obj(file_path,obj):
    dir_name=os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path,'w') as f:
        pickle.dumps(obj,f)