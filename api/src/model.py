from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, RMSprop
from constant import *
from keras.applications import VGG16, ResNet101, VGG19, MobileNetV2, Xception
from keras.regularizers import l2

def build_model():
    """
    Özel bir evrişimli sinir ağı modeli oluşturur.
    """
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def build_model_VGG16():
    """
    VGG16 modelini kullanarak bir evrişimli sinir ağı modeli oluşturur.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def build_model_MobileNet():
    """
    MobileNetV2 modelini kullanarak bir evrişimli sinir ağı modeli oluşturur.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE, pooling=None, classifier_activation="softmax")

    for layer in base_model.layers[:-10]:
        layer.trainable = True
        
    model = Sequential()
    model.add(base_model)
    model.add(GlobalMaxPooling2D())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(256, activation='relu'))  
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(len(CATEGORIES), activation='softmax')) 

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def build_model_MobileNet_RMSprop():
    """
    #MobileNetV2 modelini kullanarak RMSprop optimizer ile bir evrişimli sinir ağı modeli oluşturur.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE, pooling=None, classifier_activation="softmax")

    for layer in base_model.layers[:-10]:
        layer.trainable = True
        
    model = Sequential()
    model.add(base_model)
    model.add(GlobalMaxPooling2D())
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    
    model.add(Dense(len(CATEGORIES), activation='softmax')) 

    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def build_model_Xception():
    """
    Xception modelini kullanarak bir evrişimli sinir ağı modeli oluşturur.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE, pooling=None)
    
    for layer in base_model.layers[:-10]:
        layer.trainable = True
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  
    model.add(Dense(len(CATEGORIES), activation='softmax')) 

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model
