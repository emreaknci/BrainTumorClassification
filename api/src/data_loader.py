import os
import cv2
import numpy as np
from keras.utils import to_categorical

def load_data(DATADIR, CATEGORIES, IMG_SIZE):
    training_data = []

    def create_training_data():
        """
        Verilen kategoriler ve veri dizini temel alınarak eğitim verisini oluşturur.
        Her bir görüntüyü gri tonlamalı olarak okur, belirtilen boyuta yeniden boyutlandırır ve eğitim verisine ekler.
        """
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass

    create_training_data()

    # Veriyi X ve y olarak ayırma
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    # Veriyi uygun formatta düzenleme
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Gri tonlama kanalını ekleyin
    X = X / 255.0

    # VGG16 giriş şeklini elde etmek için gri tonlama kanalını üç kez tekrarlar
    X = np.repeat(X, 3, axis=-1)

    # Etiketleri kategorik hale getirme
    y = to_categorical(y, num_classes=4)

    return X, y
