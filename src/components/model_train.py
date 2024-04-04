import os,sys
import pickle
import pandas as pd 
import numpy as np 
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,BaggingRegressor,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_obj,evluation_model

@dataclass
class ModelTrainConfig:
    model_path : str=os.path.join('model/model.pkl')

class ModelTrain:
    def __init__(self)-> None:
        self.model_train_config=ModelTrainConfig()

    def initiate_model_train(self,train_arr,test_arr):
        try:
            logging.info('model train starts')
            x_train,y_train,x_test,y_test=(
                                    train_arr[:,:-1],
                                    train_arr[:,-1],
                                    test_arr[:,:-1],
                                    test_arr[:,-1])
            
            logging.info('Data collected')
            models = {
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression(),
                'Adaboost': AdaBoostClassifier(),
                'gradient boost': GradientBoostingClassifier(),
                'Gaussian': GaussianNB(),
                'Desision Tree':DecisionTreeClassifier()
                }

            report:dict=evluation_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            print(report)
            logging.info(f'Model Report : {report}')

            print('=======================================')
            
             # To get best model score from dictionary 
            best_model_score = max(sorted(report.values()))

            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')

            save_obj(
                obj=best_model,
                file_path=self.model_train_config.model_path
            )



        except Exception as e:
            logging.info(f'Error in Model Train {str(e)}')        
