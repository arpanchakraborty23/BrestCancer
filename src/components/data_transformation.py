import os,sys
import pickle
import pandas as pd 
import numpy as np 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import load_obj

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocess_path:str=os.path.join('preprocess/preprocess.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.Data_transformation_config=DataTransformationConfig()

    def get_data_transormation(self):
        try:
            preprocesser=Pipeline(
                steps=[
                    ('Impute',SimpleImputer(strategy='mean')),
                    ('Scale',StandardScaler())
                
                ]
            )
            
            return preprocesser
        except Exception as e:
            logging.info('Error in Data transormer')
            raise CustomException(sys,e)    
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('data transformation started')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f'data read completed {train_df.columns}')
            
            Target_col='outcome'

           
            # x_train
            input_feature_train_df=train_df.drop(columns=Target_col,axis=1)
            logging.info(f' {input_feature_train_df.columns}')
               
                # y_train
                
            target_feature_train_df=train_df[Target_col]

                # x_test
            input_feature_test_df=test_df.drop(columns=Target_col,axis=1)
                #y_test

            target_feature_test_df=test_df[Target_col]

            ## preprocessing data
            scaler=self.get_data_transormation()

            transform_input_feature_train_df=scaler.fit_transform( input_feature_train_df)
            transform_input_feature_test_df=scaler.transform( input_feature_test_df)

            train_arr=np.c_[transform_input_feature_train_df,np.array(target_feature_train_df)]
            test_arr=np.c_[transform_input_feature_test_df,np.array( input_feature_test_df)]

            print(train_arr)

            load_obj(
                file_path=self.Data_transformation_config.preprocess_path,
                obj=scaler

            )
            logging.info("Exited initiate_data_transformation method of DataTransformation class")

            return (
                train_arr,
                test_arr,
                self.Data_transformation_config.preprocess_path
            )


        except Exception as e:
            logging.info(f'Error in Data transormeration {str(e)}')
            raise CustomException(sys,e) 