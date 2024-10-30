import pyshark
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Загрузка нормального трафика из CSV-файла
normal_df = pd.read_csv('network_traffic.csv')

# Предварительная обработка данных: выбор нужных признаков
normal_df = normal_df[['Source', 'Destination', 'Protocol', 'Length', 'Time']]
normal_df['label'] = 0  # Метка 0 для нормального трафика

# Переименовываем колонки для удобства использования в модели
normal_df.rename(columns={'Source': 'src_ip', 'Destination': 'dst_ip', 'Length': 'length', 'Time': 'time'}, inplace=True)

# Преобразуем протокол в числовое значение, так как модель ожидает числовые данные
protocol_mapping = {protocol: idx for idx, protocol in enumerate(normal_df['Protocol'].unique())}
normal_df['protocol'] = normal_df['Protocol'].map(protocol_mapping)

# Импорт подозрительного трафика (DDoS)
ddos_capture = pyshark.FileCapture('traffic.pcap')
ddos_data = []
for packet in ddos_capture:
    if hasattr(packet, 'ip'):
        ddos_data.append({
            'src_ip': packet.ip.src,
            'dst_ip': packet.ip.dst,
            'protocol': packet.highest_layer,
            'length': int(packet.length),
            'time': float(packet.sniff_time.timestamp())
        })
ddos_df = pd.DataFrame(ddos_data)
ddos_df['label'] = 1  # Метка 1 для подозрительного трафика

# Объединение нормального и подозрительного трафика
full_df = pd.concat([normal_df, ddos_df], ignore_index=True)

# Нормализация данных
features = ['length', 'time']
scaler = MinMaxScaler()
full_normalized = scaler.fit_transform(full_df[features])
labels = full_df['label']

# Создание временных последовательностей для LSTM
def create_sequences(data, labels, time_steps=5):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        sequence_labels.append(labels[i + time_steps])
    return np.array(sequences), np.array(sequence_labels)

time_steps = 5
X, y = create_sequences(full_normalized, labels, time_steps)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создание модели LSTM
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(time_steps, len(features))))
model.add(Dense(1, activation='sigmoid'))  # Классификация на два класса
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Визуализация процесса обучения
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend(['Обучение', 'Валидация'])
plt.show()

# Сохранение модели
model.save('model_lstm.h5')