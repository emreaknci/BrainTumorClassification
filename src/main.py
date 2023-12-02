from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model
from train import train_model


def split_data(X, y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("x_train shape", X_train.shape)
    print("x_test shape", X_val.shape)
    print("y_train shape", Y_train.shape)
    print("y_test shape", Y_val.shape)

    return X_train, X_val, Y_train, Y_val

def plot_history(history):
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
    
    
    
DATADIR = "../data/Training"
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 150

X, y = load_data(DATADIR, CATEGORIES, IMG_SIZE)
X_train, X_val, Y_train, Y_val = split_data(X, y)
model = build_model()
history = train_model(model, X_train, Y_train, X_val, Y_val)
plot_history(history)