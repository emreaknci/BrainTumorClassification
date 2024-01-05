import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
from constant import *

def load_test_data(TESTDATADIR, img_size):
    """
    Test verisini yükleyen bir fonksiyon.
    """
    test_data = []

    def create_test_data():
        for category in os.listdir(TESTDATADIR):
            path = os.path.join(TESTDATADIR, category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # Renkli (RGB) olarak görüntü yükleme
                    new_array = cv2.resize(img_array, (img_size, img_size))
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

    X_test = np.array(X_test)
    X_test = X_test / 255.0

    return X_test, y_test, file_paths

def predict_tumor_category(model_path, X_test, y_test, file_paths, categories):
    """
    Tümör kategorisini tahmin eden bir fonksiyon.
    """
    # Modeli yükle
    model = load_model(model_path)

    # Test seti üzerinde tahminler yap
    predictions = model.predict(X_test)

    # Tahminleri sınıf etiketlerine dönüştür
    predicted_labels = [categories[np.argmax(prediction)] for prediction in predictions]

    # Dosya yolu, gerçek etiket ve tahmin edilen etiketi yazdır
    # for i in range(len(y_test)):
    #     print(f"File: {file_paths[i]}, Actual: {y_test[i]}, Predicted: {predicted_labels[i]}")

    # Gerçek etiketleri sayısal formata dönüştürme
    y_test_numeric = [categories.index(label) for label in y_test]

    # Doğruluk hesaplama
    accuracy = accuracy_score(y_test_numeric, np.argmax(predictions, axis=1))
    print(f"Accuracy for 'predict_tumor_category': {accuracy * 100:.2f}%")
    return f"Accuracy for 'predict_tumor_category': {accuracy * 100:.2f}%"
                   
def predict_tumor_status(model_path, X_test, y_test, file_paths):
    """
    Tümör durumunu tahmin eden bir fonksiyon.
    """
    # Modeli yükle
    model = load_model(model_path)

    # Test seti üzerinde tahminler yap
    predictions = model.predict(X_test)

    # Tahmini tümör durumuna dönüştürme (1 tümör varsa, 0 yoksa)
    tumor_status_predictions = [1 if prediction in [0, 1, 3] else 0 for prediction in np.argmax(predictions, axis=1)]

    # Dosya yolu, gerçek etiket ve tahmin edilen tümör durumu yazdır
    # for i in range(len(y_test)):
    #     print(f"File: {file_paths[i]}, Actual: {y_test[i]}, Predicted Tumor Status: {tumor_status_predictions[i]}")

    # Tahmini tümör durumunu (1 tümör varsa, 0 yoksa) sayısal formata dönüştürme
    y_test_tumor_status = [1 if label in ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"] else 0 for label in y_test]

    # Doğruluk hesaplama
    accuracy = accuracy_score(y_test_tumor_status, tumor_status_predictions)
    print(f"Accuracy for 'predict_tumor_status': {accuracy * 100:.2f}%")
    return f"Accuracy for 'predict_tumor_status': {accuracy * 100:.2f}%"
  
def predict_single_data_category(model_path, img_path, img_size, categories):
    """
    Tek bir veri noktasının tümör kategorisini tahmin eden bir fonksiyon.
    """
    # Görüntüyü oku ve işle
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Renkli (RGB) olarak görüntü yükleme
    new_array = cv2.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    new_array = np.array(new_array).reshape(-1, img_size, img_size, 3)  # RGB için 3 kanal kullanma

    # Modeli yükle
    model = load_model(model_path)

    # Tahmin yap
    prediction = model.predict(new_array)

    # Tahmini sınıf etiketine dönüştürme
    predicted_label = categories[np.argmax(prediction)]

    tumor_type = LABEL_MESSAGES[predicted_label]
    return f"{tumor_type}'"

def predict_single_data_status(model_path, img_path, img_size):
    """
    Tek bir veri noktasının tümör durumunu tahmin eden bir fonksiyon.
    """
    # Görüntüyü oku ve işle
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Renkli (RGB) olarak görüntü yükleme
    new_array = cv2.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    new_array = np.array(new_array).reshape(-1, img_size, img_size, 3)  # RGB için 3 kanal kullanma

    # Modeli yükle
    model = load_model(model_path)

    # Tahmin yap
    prediction = model.predict(new_array)

    # Tahmini tümör durumuna dönüştürme (1 tümör varsa, 0 yoksa)
    tumor_status_prediction = 1 if np.argmax(prediction) in [0, 1, 3] else 0

    if tumor_status_prediction == 1:
        return True
    if tumor_status_prediction == 0:
        return False

if __name__ == "__main__":
    X_test, y_test, file_paths = load_test_data(TEST_DATA_DIR, IMG_SIZE)

    # print("--------------------------")
    # print(("ilk model"))
    
    # predict_tumor_category(MODEL_DIR, X_test, y_test, file_paths, CATEGORIES)
    # predict_tumor_status(MODEL_DIR, X_test, y_test, file_paths)
    
    print("--------------------------")
    print(("vgg16"))
    predict_tumor_category(VGG16, X_test, y_test, file_paths, CATEGORIES)
    predict_tumor_status(VGG16, X_test, y_test, file_paths)
    
    print("--------------------------")
    print(("mobileNet"))
    predict_tumor_category(MOBILE_NET, X_test, y_test, file_paths, CATEGORIES)
    predict_tumor_status(MOBILE_NET, X_test, y_test, file_paths)


    print("--------------------------")
    print(("xception"))
    predict_tumor_category(XCEPTION, X_test, y_test, file_paths, CATEGORIES)
    predict_tumor_status(XCEPTION, X_test, y_test, file_paths)

    print("--------------------------")
    print(("MOBILE_NET_RMSprop"))
    predict_tumor_category(MOBILE_NET_RMSprop, X_test, y_test, file_paths, CATEGORIES)
    predict_tumor_status(MOBILE_NET_RMSprop, X_test, y_test, file_paths)
