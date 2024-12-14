import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error

# Przygotowanie danych
dataset = pd.read_csv('data/airline-passengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset.set_index(['Month'], inplace=True)

plt.figure(figsize=(16, 9))
plt.plot(dataset['Passengers'])
plt.show()

train_size = int(len(dataset) * 0.70)
scaler = MinMaxScaler(feature_range=(0, 1))


def create_dataset(df, train_size):
    test_size = len(df) - train_size
    train, test = df[0:train_size, :].copy(), df[train_size:len(df), :].copy()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Tworzenie zbioru treningowego
    for i in range(len(train) - 12 - 1):
        X_train.append(train[i:(i + 12), 0])
        y_train.append(train[i + 12, 0])

    # Tworzenie zbioru testowego
    for i in range(len(test) - 12 - 1):
        X_test.append(test[i:(i + 12), 0])
        y_test.append(test[i + 12, 0])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


X_train, y_train, X_test, y_test = create_dataset(np.array(dataset), train_size)

# Rozdzielanie na dane treningowe, walidacyjne i testowe
train_val_size = int(0.8 * len(X_train))
X_train, X_val = X_train[:train_val_size], X_train[train_val_size:]
y_train, y_val = y_train[:train_val_size], y_train[train_val_size:]


# Funkcja backtestingu z siatką hiperparametrów
def backtest(model_type, look_back_range=range(1, 13), units_range=range(1, 13)):
    best_model = None
    best_rmse = float('inf')

    for look_back in look_back_range:
        for units in units_range:
            model = Sequential()

            if model_type == 'RNN':
                model.add(SimpleRNN(units, input_shape=(look_back, 1)))
            elif model_type == 'LSTM':
                model.add(LSTM(units, input_shape=(look_back, 1)))
            elif model_type == 'GRU':
                model.add(GRU(units, input_shape=(look_back, 1)))

            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_val, y_val), verbose=0)

            rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_val.reshape(-1, 1)),
                                              scaler.inverse_transform(model.predict(X_val).reshape(-1, 1))))

            print(f"{model_type} - Look_back: {look_back}, Units: {units}, RMSE: {rmse}")

            if rmse < best_rmse:
                best_model = (model, look_back, units)
                best_rmse = rmse

    return best_model


# Wykonanie backtestingu dla każdej architektury
models = {
    'RNN': backtest('RNN'),
    'LSTM': backtest('LSTM'),
    'GRU': backtest('GRU')
}

# Zwizualizacja wyników najlepszego modelu
best_model, look_back, units = models[max(models, key=models.get)]
print(f"\nBest model: {best_model[0]}")
print(f"Look_back: {look_back}")
print(f"Units: {units}")

predictions = best_model[0].predict(X_test)
predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(16, 9))
plt.plot(actual_values, color='blue', label='True values')
plt.plot(predicted_values, color='green', label='Prediction')
plt.legend(loc='upper left')
plt.title(f'Prediction vs True Values - Best {best_model[0]} Model')
plt.show()
