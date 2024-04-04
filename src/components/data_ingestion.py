import os,sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_path:str=os.path.join('artifacts','train.csv')
    test_path:str=os.path.join('artifacts','test.csv')
    raw_path:str=os.path.join('artifacts','raw.csv')

class Dataungestion:
    def __init__(self) -> None:
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self)-> None:
        try:
            logging.info('data ingestion has started')
            data=load_breast_cancer()
            df=pd.DataFrame(data.data,columns=data.feature_names)
            df['outcome']=data.target

            logging.info(f'data ingestion has started {df.head()}')

            # os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            # df.to_csv(self.data_ingestion_config.raw_data_path)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_path)

            train_path,test_path=train_test_split(df,test_size=0.26,random_state=40)
            train_path.to_csv(self.data_ingestion_config.train_path,index=False)

            test_path.to_csv(self.data_ingestion_config.test_path,index=False)

            logging.info('artifacts created')
            logging.info('data ingestion completed')
            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )

        except Exception as e:
            logging.info(f'Error occured {str(e)}')
            raise CustomException(sys,e)    
        
# if __name__=='__main__':
#     obj=Dataungestion()
#     train_path,test_path=obj.initiate_data_ingestion()        

#     transform_obj=DataTransformation()
#     train_arr,test_arr,_=transform_obj.initiate_data_transformation(train_path,test_path)

#     model=ModelTrain()
#     print(model.initiate_model_train(train_arr,test_arr))