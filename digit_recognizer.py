import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    x_train,
    y_train,
    epochs=5
)


loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

prediction = model.predict(
    x_test[:1]
)

predicted_digit = prediction.argmax()

print(
    "Predicted Digit:",
    predicted_digit
)

print(
    "Actual Digit:",
    y_test[0]
)


plt.imshow(
    x_test[0],
    cmap="gray"
)

plt.title(
    f"Prediction: {predicted_digit}"
)

plt.show()


model.save(
    "mnist_model.keras"
)

print(
    "Model saved successfully!"
)