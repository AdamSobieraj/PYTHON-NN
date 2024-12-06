import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Wczytanie danych z pliku iris.csv
iris = pd.read_csv('data/iris.csv')

# Zamiana etykiet tekstowych na numeryczne
label_encoder = LabelEncoder()
iris['variety'] = label_encoder.fit_transform(iris['variety'])

# Podział na cechy i etykiety
X = iris.drop('variety', axis=1).values
y = iris['variety'].values

# Normalizacja cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Konwersja etykiet do reprezentacji one-hot
y = tf.keras.utils.to_categorical(y)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definicja modelu
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 klasy
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Ewaluacja na zbiorze testowym
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Dokładność na zbiorze testowym: {test_accuracy * 100:.2f}%")

# Wykresy strat i dokładności
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.show()

# Przykładowe przewidywanie
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Przykład Setosa
sample = scaler.transform(sample)  # Normalizacja
prediction = model.predict(sample)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print(f"Przewidywany gatunek: {predicted_class[0]}")
