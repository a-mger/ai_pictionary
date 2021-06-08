# Keras needs images whose last dimension is the number of channels, its 1 for black and white.

from tensorflow.keras import layers
from tensorflow.keras import models


def initialize_model(X_train, label_names):
    model = models.Sequential()
    model.add(
        layers.Convolution2D(16, (3, 3),
                             padding='same',
                             input_shape=X_train.shape[1:],
                             activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(label_names), activation='softmax'))
    # Train model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['top_k_categorical_accuracy'])
    return model
