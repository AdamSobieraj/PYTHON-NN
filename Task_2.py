import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Krok 1: Przygotowanie danych i budowa modelu
# Dane
(images, labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizacja danych
images = images / 255.0
images = images.reshape(-1, 28, 28, 1)  # Dodanie kanału
labels = to_categorical(labels, 10)  # One-hot encoding

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=10, stratify=np.argmax(labels, axis=1)
)

# Budowa modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Ocena modelu
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Zapisanie modelu
model.save("fashion_mnist_model.h5")


# Krok 2: Interfejs użytkownika
def predict_and_visualize(model_path, image_index):
    # Wczytaj zapisany model
    model = load_model(model_path)

    # Wybranie obrazu do predykcji
    image = X_test[image_index]
    label = np.argmax(y_test[image_index])  # Prawidłowa etykieta

    # Przetworzenie obrazu do predykcji
    image_for_prediction = np.expand_dims(image, axis=0)

    # Predykcja
    prediction = np.argmax(model.predict(image_for_prediction), axis=1)[0]

    # Wyświetlenie wyniku
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {prediction}, True: {label}")
    plt.axis('off')
    plt.show()


# Wywołanie funkcji dla przykładowego obrazu
predict_and_visualize("fashion_mnist_model.h5", image_index=0)

# Krok 3: Augmentacja danych i ulepszenie modelu
# Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

# Generowanie danych na podstawie X_train
datagen.fit(X_train)

# Budowa nowego modelu
model_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Kompilacja modelu
model_aug.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Trenowanie modelu z augmentacją
history_aug = model_aug.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(X_test, y_test)
)

# Ocena nowego modelu
test_loss_aug, test_accuracy_aug = model_aug.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy after augmentation: {test_accuracy_aug:.4f}")

# Zapisanie ulepszonego modelu
model_aug.save("fashion_mnist_model_augmented.h5")
