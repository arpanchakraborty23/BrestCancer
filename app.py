from flask import Flask,render_template,request,jsonify,send_file
import sys
from src.exception import CustomException
from src.logger import logging
from src.pipline.pradiction_pipline import Prediction,PredictionConfig

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method=='POST':
            pipline=Prediction(request)

            prediction_file_detail=pipline.run_pipline()
            return send_file(path_or_file=pipline.prediction_config.prediction_file_path,
                            download_name= pipline.prediction_config.prediction_file_name,
                            as_attachment= True)
             
            
        else:
            return render_template('upload.html')
    except Exception as e:
        logging.info(f'error in bulk app prediction {str(e)}')
        raise CustomException(sys,e) from e         
    
if __name__=='__main__':
        app.run(port=5000,debug=True)