import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.keras.utils.set_random_seed(42)

conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

dummy_input = tf.random.normal((1, 128, 128, 1))

output = conv(dummy_input)

weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)

print("İlk filtre:")
print(weights[:, :, 0, 0])

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(weights[:, :, 0, i], cmap="gray")
    plt.title(f"Filtre {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("filtreler.png")
plt.show()