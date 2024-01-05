from keras.preprocessing.image import ImageDataGenerator
from constant import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=64):
    """
    Modeli eğitmek için kullanılan fonksiyon.
    
    Args:
        model (keras.Model): Eğitilecek model.
        X_train (numpy.ndarray): Eğitim verisi.
        Y_train (numpy.ndarray): Eğitim etiketleri.
        X_val (numpy.ndarray): Doğrulama verisi.
        Y_val (numpy.ndarray): Doğrulama etiketleri.
        epochs (int): Eğitim epoch sayısı.
        batch_size (int): Minibatch boyutu.

    Returns:
        keras.callbacks.History: Eğitim geçmişi.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Veri artırma işlemlerini uygula
    datagen.fit(X_train)
    
    # Erken durdurma, öğrenme oranını azaltma ve öğrenme oranı zamanlayıcısı gerçekleştir
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    def lr_schedule(epoch):
        initial_lr = 0.001
        drop_factor = 0.1
        epochs_drop = 10
        return initial_lr * (drop_factor ** (epoch // epochs_drop))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Modeli eğit
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, 
                        validation_data=(X_val, Y_val),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        callbacks=[early_stopping, reduce_lr, lr_scheduler]
                        )

    # Eğitilmiş modeli kaydet
    model.save(f"{MODEL_DIR}/model.h5")
    
    return history
