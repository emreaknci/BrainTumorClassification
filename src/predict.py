import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

def load_test_data(TESTDATADIR, IMG_SIZE):
    test_data = []

    def create_test_data():
        for category in os.listdir(TESTDATADIR):
            path = os.path.join(TESTDATADIR, category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    test_data.append([new_array, os.path.join(path, img), category])
                except Exception as e:
                    pass

    create_test_data()

    X_test = []
    y_test = []
    file_paths = []
    for features, file_path, label in test_data:
        X_test.append(features)
        file_paths.append(file_path)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE)
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X_test, y_test, file_paths

def predict_tumor_category(model, X_test, y_test, file_paths, categories):
    # Assuming you already have a trained model
    model = load_model("../models/model.h5")

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Convert predictions to class labels
    predicted_labels = [categories[np.argmax(prediction)] for prediction in predictions]

    # Print the file path, actual label, and predicted label
    # for i in range(len(y_test)):
    #     print(f"File: {file_paths[i]}, Actual: {y_test[i]}, Predicted: {predicted_labels[i]}")

    # Convert true labels to numeric format
    y_test_numeric = [categories.index(label) for label in y_test]

    # Calculate accuracy
    accuracy = accuracy_score(y_test_numeric, np.argmax(predictions, axis=1))
    print(f"Accuracy for 'predict_tumor_category': {accuracy * 100:.2f}%")
                   
def predict_tumor_status(model, X_test, y_test, file_paths):
    # Assuming you already have a trained model
    model = load_model("../models/model.h5")

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Convert predictions to tumor status (tumor var: 1, tumor yok: 0)
    tumor_status_predictions = [1 if prediction in [0, 1, 3] else 0 for prediction in np.argmax(predictions, axis=1)]

    # Print the file path, actual label, and predicted tumor status
    # for i in range(len(y_test)):
    #     print(f"File: {file_paths[i]}, Actual: {y_test[i]}, Predicted Tumor Status: {tumor_status_predictions[i]}")

    # Convert true labels to tumor status (tumor var: 1, tumor yok: 0)
    y_test_tumor_status = [1 if label in ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"] else 0 for label in y_test]

    # Calculate accuracy
    accuracy = accuracy_score(y_test_tumor_status, tumor_status_predictions)
    print(f"Accuracy for 'predict_tumor_status': {accuracy * 100:.2f}%")
  
def predict_single_data_category(model, img_path, IMG_SIZE, categories):
    # Read and preprocess the image
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array / 255.0
    new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Make a prediction
    prediction = model.predict(new_array)

    # Convert prediction to class label
    predicted_label = categories[np.argmax(prediction)]

    print(f"File: {img_path}, Predicted: {predicted_label}")
 
def predict_single_data_status(model, img_path, IMG_SIZE):
    # Read and preprocess the image
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array / 255.0
    new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Make a prediction
    prediction = model.predict(new_array)

    # Convert prediction to tumor status (tumor var: 1, tumor yok: 0)
    tumor_status_prediction = 1 if prediction.argmax() in [0, 1, 3] else 0

    print(f"File: {img_path}, Predicted Tumor Status: {tumor_status_prediction}")
 
if __name__ == "__main__":
    TESTDATADIR = "../data/Testing"
    IMG_SIZE = 150
    CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

    X_test, y_test, file_paths = load_test_data(TESTDATADIR, IMG_SIZE)
    model = load_model("../models/model.h5") 
    predict_tumor_category(model, X_test, y_test, file_paths, CATEGORIES)
    predict_tumor_status(model, X_test, y_test, file_paths)
    TEST_IMAGE_PATH="../data/Testing/glioma_tumor/image(2).jpg"
    
    predict_single_data_category(model, TEST_IMAGE_PATH, IMG_SIZE, CATEGORIES)
    predict_single_data_status(model, TEST_IMAGE_PATH, IMG_SIZE)

