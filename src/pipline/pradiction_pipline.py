import os,sys
import pickle
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from flask import request

from src.utils.utils import load_obj,insert_data_db
from dotenv import load_dotenv
load_dotenv()

db=os.getenv('db_prediction')
collection=os.getenv('input_csv')
prediction_csv=os.getenv('prediction')


from dataclasses import dataclass

@dataclass
class PredictionConfig:
    model_path=os.path.join('model/model.pkl')
    preprocesser_path=os.path.join('src\preprocess\preprocess.pkl')

    dir_name="predictions"
    prediction_file_name="predicted_file.csv"
    prediction_file_path=os.path.join(dir_name,prediction_file_name)

class Prediction:
    def __init__(self,request) -> None:
        self.prediction_config=PredictionConfig()
        self.request=request

    def save_input_files(self):
        try:
            logging.info('save files 27')
            # predicion  dir
            input_file_dir='prediction_artifats'
            os.makedirs(input_file_dir,exist_ok=True)

            input_csv_file=request.files['file']
            pred_file_path=os.path.join(input_file_dir,input_csv_file.filename)

            # save files
            input_csv_file.save(pred_file_path)
           
           
            logging.info('save files com 38')

            return pred_file_path


        except Exception as e:
            logging.info('error ',str(e))
            raise CustomException(sys,e)  

    def predict(self,feature):
        try:
            model=load_obj(self.prediction_config.model_path)
            scaler=load_obj(self.prediction_config.preprocesser_path)

            # scale data
            scale_data=scaler.transform(feature)

            pred=model.predict(scale_data)

            return pred
        except Exception as e:
            logging.info('error ',str(e))
            raise CustomException(sys,e)  


    def get_prediction_as_df(self,input_df_path:pd.DataFrame):
        try:
            input_df=pd.read_csv(input_df_path)
            

            Target_col='outcome'

           
            input_df = input_df.drop(columns=['Unnamed: 0'],axis=1) if 'Unnamed: 0' in input_df.columns else input_df

           
            input_df =  input_df.drop(columns=Target_col,axis=1) if Target_col in input_df.columns else input_df
            
            # insert_data_db(database=db,collection=collection,df=input_df)

            
            #prediction
            pred=self.predict(feature=input_df)
            # prediction col
            input_df[Target_col]=[i for i in pred]

            #map col
            map_col={0:'B',1:'M'}

            # input_df[Target_col]=input_df.map(map_col)
            os.makedirs( self.prediction_config.dir_name, exist_ok= True)
            input_df.to_csv(self.prediction_config.prediction_file_path, index= False)
            logging.info('prediction completed')

           
            # logging.info(f'prediction df : {insert_data_db(database=db,collection=prediction_csv,df=input_df)}')


        except Exception as e:
            logging.info(str(e))
            raise CustomException(sys,e) 


    def run_pipline(self):
        try:
            input__csv_file=self.save_input_files()
            #data collection input file


            prediction=self.get_prediction_as_df(input_df_path=input__csv_file)

            
           


            return prediction

        except Exception as e:
                logging.info(f'error {str(e)}')
                raise CustomException(sys,e)

