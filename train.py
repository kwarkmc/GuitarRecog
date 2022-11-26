import keras.utils
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['c', 'g', 'f']

data = np.concatenate([
    np.load('dataset/seq_c_1669411456.npy'),
    np.load('dataset/seq_f_1669411456.npy'),
    np.load('dataset/seq_g_1669411456.npy')
], axis=0)

#print(data.shape) : (762, 30, 100)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

#print(x_data.shape) : 762, 30, 99
#print(labels.shape) : 762,

y_data = keras.utils.to_categorical(labels, num_classes = len(actions))
#print(y_data.shape) : 762, 3

from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.1, random_state = 2022)

#print(x_train.shape, y_train.shape) : (685, 30, 99), (685, 3)
#print(x_val.shape, y_val.shape) : (77, 30, 99), (77, 3)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16,10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train_acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.python.keras.models import load_model

model = load_model('models/model.h5')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))