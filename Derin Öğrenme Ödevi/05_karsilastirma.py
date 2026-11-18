import tensorflow as tf
import numpy as np
import time


(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


def deney_yap(filtre_sayisi):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    print(f"\n===== {filtre_sayisi} FİLTRELİ MODEL =====")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            filtre_sayisi, (3, 3), activation='relu'
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu'
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    baslangic = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    egitim_suresi = time.perf_counter() - baslangic

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test
    )

    egitim_dogrulugu = history.history['accuracy'][-1]

    return {
        "filtre": filtre_sayisi,
        "egitim": egitim_dogrulugu,
        "test": test_accuracy,
        "sure": egitim_suresi,
        "parametre": model.count_params()
    }



deney1 = deney_yap(8)
deney2 = deney_yap(16)


print("\n===== KARŞILAŞTIRMA SONUÇLARI =====")

for deney in [deney1, deney2]:

    print(f"\nFiltre sayısı: {deney['filtre']}")
    print(f"Eğitim doğruluğu: %{deney['egitim'] * 100:.2f}")
    print(f"Test doğruluğu: %{deney['test'] * 100:.2f}")
    print(f"Eğitim süresi: {deney['sure']:.2f} saniye")
    print(f"Toplam parametre: {deney['parametre']}")