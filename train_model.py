import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

os.makedirs("model", exist_ok=True)

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32")  / 255.0

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print(f"Training samples: {len(x_train)}")
print(f"Test samples:     {len(x_test)}")

# -------------------------------------------------------
# DATA AUGMENTATION
# This creates slightly modified versions of training images:
# - randomly rotated (up to 15 degrees)
# - randomly zoomed in/out
# - randomly shifted left/right/up/down
# This teaches the model to handle real hand-drawn input
# -------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

print("\nBuilding improved CNN model...")

model = models.Sequential([
    # Apply augmentation only during training
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Learning rate scheduler: starts fast, slows down as training progresses
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

# Early stopping: stops training if accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining with data augmentation (takes longer but much more accurate)...")

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("model/digit_recognizer.keras")
print("\nImproved model saved!")