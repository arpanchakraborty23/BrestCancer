import mlflow
import mlflow.sklearn
import numpy as np
import pickle
import os,sys
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
load_dotenv()

from src.utils.utils import load_obj
from urllib.parse import urlparse
from src.logger import logging
from src.exception import CustomException

class ModelEvaluation:
    def eval_metrics(self,actual,pred):
        accuracy=accuracy_score(actual,pred)*100

        confusion=confusion_matrix(actual,pred)
        report=classification_report(actual,pred)
        logging.info('eval metrix capture')

        return accuracy,confusion,report
    
    def initiate_model_eval(self,train_arr,test_arr):
        try:    

            logging.info('saparate data to test data')
            x_test,y_test=(test_array[:,:-1],test_array[:,-1])

            model_path=os.path.join('model/model.pkl')
            model=load_obj(model_path)

            # register
            MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            logging.info('model has register')
            
            #start server
            with mlflow.start_run():
                prediction=model.predict(x_test)
                accuracy,confusion,report=self.eval_metrics(actual=y_test,pred=prediction)

                mlflow.log_metric('Confusion metrix', confusion)
                mlflow.log_metric('classification report',report )
                mlflow.log_metric('accuracy',accuracy)

                if tracking_url_type_store != "file":

                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                
                    mlflow.sklearn.log_model(model, "model")

                    logging.info('completed model eval')
        except Exception as e:
            logging.info(f' Error occured {str(e)}')
            raise CustomException(sys,e)
            


