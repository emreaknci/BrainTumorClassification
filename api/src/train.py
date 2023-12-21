from keras.preprocessing.image import ImageDataGenerator
from constant import *
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=40):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(X_train)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    def lr_schedule(epoch):
        initial_lr = 0.001
        drop_factor = 0.1
        epochs_drop = 10
        return initial_lr * (drop_factor ** (epoch // epochs_drop))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, 
                        validation_data=(X_val, Y_val),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        callbacks=[early_stopping, reduce_lr, lr_scheduler]
                        )

    model.save(f"{MODEL_DIR}/model.h5")
    
    return history

