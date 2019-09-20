from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D, InputLayer, Flatten, Reshape
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def createCDNet(summary=False, output_layers=8):
    print("Start Initialzing Neural Network!")
    model = Sequential()

    model.add(InputLayer(input_shape=(2, 75)))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    # model.add(BatchNormalization())

    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(output_layers, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    if summary:
        print(model.summary())
    return model

def scale_CD_data(data, vmin, vmax):
    return (data / vmax - vmin / vmax) / (1 - vmin / vmax)

data = pd.read_csv('fima.txt')

data = data.values

new_spectrum = []

for n, row in enumerate(data):

    #remove half the nm
    if n%2 == 0:
        new_spectrum.append(row)

while np.max(np.array(new_spectrum)[:,0]) < 265:
    new_spectrum.append(np.array([np.max(np.array(new_spectrum)[:,0])+1, 0.0]))

plot = np.array(new_spectrum)

new_spectrum = np.reshape(np.expand_dims(np.array(new_spectrum[1:]), axis=0), (-1,2,75))

new_spectrum[:,1] = scale_CD_data(new_spectrum[:,1], np.min(new_spectrum[:,1]), np.max(new_spectrum[:,1]))
print(new_spectrum.shape)




model = createCDNet(output_layers=7)
model.load_weights('weights-improvement-177-1.00.hdf5')

prediction = model.predict(new_spectrum)
print(prediction)

y_data = np.array([0.26168224, 0.07476636, 0.105919,   0.01246106, 0.0623053,  0.46417445,
 0.01869159, 0.        ])

y_data = y_data[0], y_data[1], y_data[2], y_data[4], y_data[6], y_data[7], y_data[3] + y_data[5]

print(y_data)