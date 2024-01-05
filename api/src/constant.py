import os

# Veri dizini oluşturma
DATA_DIR = "../data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Kategorilerin tanımlanması
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224

# Eğitim ve test verisi dizinleri
TRAINING_DATA_DIR = "../data/Training"
TEST_DATA_DIR = "../data/Testing"

# Model dizini
MODEL_DIR = "../models"
MOBILE_NET = "../models/MobileNetV2/modelMobileNet.h5"
MOBILE_NET_RMSprop = "../models/MobileNetV2_RMSprop/model.h5"
VGG16 = "../models/VGG16/modelVGG16.h5"
XCEPTION = "../models/Xception/xception.h5"
FIRST_MODEL = "../models/model/model.h5"

# Eğitim parametreleri
BATCH_SIZE = 40
EPOCHS = 50

# Test verisinin oranı
TEST_SIZE = 0.2

# Giriş şekli
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Sonuç mesajı
MESSAGE = "Ancak sonuçlar tamamen doğru olmayabilir, bu nedenle kesin bir cevap için bir profesyonelden yardım almak iyi bir fikir olabilir."

# Etiket mesajları
LABEL_MESSAGES = {
    "pituitary_tumor": "Hipofiz Tümörü",
    "meningioma_tumor": "Meningiyoma Tümörü",
    "glioma_tumor": "Glioma Tümörü",
    "no_tumor": "Tümör Yok"
}
