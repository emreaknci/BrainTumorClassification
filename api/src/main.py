from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import *
from train import train_model
from constant import * 

def split_data(X, y):
    """
    Veriyi eğitim ve test setlerine bölen bir fonksiyon.
    """
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    print("x_train shape", X_train.shape)
    print("x_test shape", X_val.shape)
    print("y_train shape", Y_train.shape)
    print("y_test shape", Y_val.shape)

    return X_train, X_val, Y_train, Y_val

def plot_history(history):
    """
    Eğitim geçmişini görselleştiren bir fonksiyon.
    """
    plt.plot(history.history["loss"], c="purple")
    plt.plot(history.history["val_loss"], c="orange")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "test"])
    plt.show()
    
    plt.plot(history.history["accuracy"], c="purple")
    plt.plot(history.history["val_accuracy"], c="orange")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train", "test"])
    plt.show()

# Veriyi yükleme
X, y = load_data(TRAINING_DATA_DIR, CATEGORIES, IMG_SIZE)

# Veriyi eğitim ve test setlerine böleme
X_train, X_val, Y_train, Y_val = split_data(X, y)

# Model oluşturma (istenen model seçeneğini seç)
# model = build_model()
# model = build_model_VGG16()
# model = build_model_MobileNet()
# model = build_model_Xception()
model = build_model_MobileNet_RMSprop()

# Modeli eğitme
history = train_model(model, X_train, Y_train, X_val, Y_val)

# Eğitim geçmişini görselleştirme
plot_history(history)
