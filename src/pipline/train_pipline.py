from src.components.data_ingestion import Dataungestion
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import CustomException

logging.info(' Train pipeline has started')
obj=Dataungestion()
train_path,test_path=obj.initiate_data_ingestion()        

transform_obj=DataTransformation()
    
train_arr,test_arr,_=transform_obj.initiate_data_transformation(train_path,test_path)

model=ModelTrain()
print(model.initiate_model_train(train_arr,test_arr))

mlflow_obj=ModelEvaluation()
print(mlflow_obj.initiate_model_eval(train_arr,test_arr))

logging.info('Train has completed')