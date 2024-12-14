import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from Base import create_dataset

# Załaduj dane
dataset = pd.read_csv('data/airline-passengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset.set_index('Month', inplace=True)

# Logarytmiczne skalowanie
dataset_log = np.log(dataset['Passengers'])

# Obliczenie różnic między kolejnymi wartościami
dataset_diff = dataset_log.diff().dropna()

# Standaryzacja danych
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset_diff)

# Przygotowanie danych wejściowych
look_back = 3
X_train, X_test, y_train, y_test = create_dataset(df=dataset_scaled, train_size=int(len(dataset_scaled) * 0.70), lback=look_back)

# Model RNN
model_rnn = Sequential()
model_rnn.add(SimpleRNN(5, input_shape=(1, look_back)))
model_rnn.add(Dense(1))
model_rnn.compile(loss='mean_squared_error', optimizer='adam')
model_rnn.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
y_pred_rnn = model_rnn.predict(X_test)
y_pred_rnn = scaler.inverse_transform(y_pred_rnn).flatten()

# Model LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(5, input_shape=(1, look_back)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
y_pred_lstm = model_lstm.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm).flatten()

# Model GRU
model_gru = Sequential()
model_gru.add(GRU(5, input_shape=(1, look_back)))
model_gru.add(Dense(1))
model_gru.compile(loss='mean_squared_error', optimizer='adam')
model_gru.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1)
y_pred_gru = model_gru.predict(X_test)
y_pred_gru = scaler.inverse_transform(y_pred_gru).flatten()

# Wizualizacja i ocena
plt.figure(figsize=(16,9))
plt.plot(dataset['Passengers'], color='blue', label='True values')
plt.plot(pd.Series(y_pred_rnn, index=dataset.index[-len(y_pred_rnn):]), color='green', label='RNN prediction')
plt.plot(pd.Series(y_pred_lstm, index=dataset.index[-len(y_pred_lstm):]), color='red', label='LSTM prediction')
plt.plot(pd.Series(y_pred_gru, index=dataset.index[-len(y_pred_gru):]), color='orange', label='GRU prediction')
plt.legend()
plt.show()

# Obliczenie RMSE
rmse_rnn = np.sqrt(mean_squared_error(dataset['Passengers'][-len(y_pred_rnn):], y_pred_rnn))
rmse_lstm = np.sqrt(mean_squared_error(dataset['Passengers'][-len(y_pred_lstm):], y_pred_lstm))
rmse_gru = np.sqrt(mean_squared_error(dataset['Passengers'][-len(y_pred_gru):], y_pred_gru))

print(f"RMSE RNN: {rmse_rnn}")
print(f"RMSE LSTM: {rmse_lstm}")
print(f"RMSE GRU: {rmse_gru}")
