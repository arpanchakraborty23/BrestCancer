import os,sys
import pickle
import pandas as pd 
import numpy as np 

from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging


def save_obj(file_path, obj):
  with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def evluation_model(x_train,y_train,x_test,y_test,models):
    try:
        logging.info(' model evaluation started')
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            
            test_model_score = accuracy_score(y_test,y_test_pred)*100

            report[list(models.keys())[i]] =  test_model_score

        return report
        
    except Exception as e:
        logging.info(f' Error {str(e)}')
        raise CustomException(sys,e)
    
def load_obj(file_path):
    with open(file_path,'rb') as f:
        obj=pickle.load(f)

        return obj


