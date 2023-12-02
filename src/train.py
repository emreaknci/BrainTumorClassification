from keras.preprocessing.image import ImageDataGenerator


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=40):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        zoom_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=True,
        vertical_flip=False
    )

    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, validation_data=(X_val, Y_val),
                        steps_per_epoch=X_train.shape[0] // batch_size)

    model.save("../models/model.h5")
    
    return history

