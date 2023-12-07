from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from predict import predict_single_data_category,predict_single_data_status
from constant import *

app = Flask(__name__)
CORS(app)

@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        # Get the image file from the POST request
        file = request.files['image']

        # Check if the file has a valid filename
        if file and allowed_file_type(file.filename):
            image_path = f"../data/{file.filename}"
            file.save(image_path)
           
            model_path=f"{MODEL_DIR}/model.h5"
            result = predict_single_data_category(model_path, image_path, IMG_SIZE, CATEGORIES)

            os.remove(image_path)

            return jsonify(type=result)

        else:
            return jsonify(error="Invalid file format. Supported formats: ['jpg', 'jpeg', 'png']")

    except Exception as e:
        return jsonify(error=str(e))
    
@app.route('/predict_status', methods=['POST'])
def predict_status():
    try:
        # Get the image file from the POST request
        file = request.files['image']

        # Check if the file has a valid filename
        if file and allowed_file_type(file.filename):
            image_path = f"../data/{file.filename}"
            file.save(image_path)
           
            model_path=f"{MODEL_DIR}/model.h5"
            result = predict_single_data_status(model_path, image_path, IMG_SIZE)

            os.remove(image_path)

            return jsonify(result=result)

        else:
            return jsonify(error="Invalid file format. Supported formats: ['jpg', 'jpeg', 'png']")

    except Exception as e:
        return jsonify(error=str(e))


def allowed_file_type(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
