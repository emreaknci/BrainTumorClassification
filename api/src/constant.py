import os

DATA_DIR="../data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224

# Path: src/data/..
TRAINING_DATA_DIR = "../data/Training"
TEST_DATA_DIR = "../data/Testing"


# Path: src/models/..
MODEL_DIR = "../models"
MOBILE_NET = "../models/MobileNetV2/modelMobileNet.h5"
VGG16 = "../models/VGG16/modelVGG16.h5"
FIRST_MODEL = "../models/model/model.h5"




BATCH_SIZE = 40
EPOCHS = 50

TEST_SIZE = 0.2

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

MESSAGE="However, the results may not be completely accurate, so it's a good idea to ask a professional for a definite answer."

LABEL_MESSAGES = {
    "pituitary_tumor": "Pituitary Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "glioma_tumor": "Glioma Tumor",
    "no_tumor": "no tumor"
}