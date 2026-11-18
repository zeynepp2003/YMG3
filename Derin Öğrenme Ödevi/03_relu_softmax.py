import tensorflow as tf


x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])

relu = tf.keras.layers.ReLU()
y = relu(x)

print("===== ReLU =====")
print("Giriş:", x.numpy())
print("ReLU çıkışı:", y.numpy())


logits = tf.constant([[2.0, 1.0, 0.1]])

softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)

print("\n===== Softmax =====")
print("Softmax sonuçları:")
print(probabilities.numpy())

print("Olasılıkların toplamı:")
print(tf.reduce_sum(probabilities).numpy())