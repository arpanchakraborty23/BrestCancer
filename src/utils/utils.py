import os,sys
import pickle
import pandas as pd 
import numpy as np
import json
from pymongo import MongoClient
from dotenv import load_dotenv 

load_dotenv()

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

def data_from_db(database,collection):
    try:
        logging.info('data read started')
        
       
        client = MongoClient(os.getenv('url'))
        
        logging.info('database')
        # Select the database and collection
        database = client[database]
        logging.info('data collection')
        collection = database[collection]

        # Query to retrieve data from the collection
        data = collection.find()

        df = pd.DataFrame(list(collection.find()))
        df.drop(columns=['_id'],axis=1,inplace=True)

        return df

    except Exception as e:
            logging.info('Error occured: ') 
            raise CustomException(sys,e) from e 
             
def insert_data_db(database,collection,df):
    try:
        client = MongoClient(os.getenv('url'))
        db=client[database]
        collection=db[collection]
        data=df.to_dict(orient='records')
        print(data)

        collection.insert_many(data)
    except Exception as e:
        logging.info(f' error {str(e)}')
        raise CustomException(sys,e)

