import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D, InputLayer, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
# from keras.preprocessing import
import matplotlib.pyplot as plt
from keras.models import load_model


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






x_data = np.load('CDnet_xdata.npy', allow_pickle=True)
y_data = np.load('CDnet_ydata.npy', allow_pickle=True)

reduced_y = np.stack(
    (y_data[:, 0], y_data[:, 1], y_data[:, 2], y_data[:, 4], y_data[:, 6], y_data[:, 7], y_data[:, 3] + y_data[:, 5]),
    axis=1)

reduced_y = np.stack(
    (y_data[:, 0], y_data[:, 3], y_data[:, 5], y_data[:, 6], y_data[:, 1]+y_data[:, 2]+y_data[:, 4]+y_data[:, 7]),
    axis=1)


y_data = reduced_y

print(reduced_y.shape)
# quit()

X_train, X_test, Y_train, Y_test = train_test_split(x_data, reduced_y, test_size=0.20)

print(X_train.shape)
print(Y_train.shape)

train = False

if train:
    model = createCDNet(True, 5)
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',
                                 save_weights_only=False)
    callbacks_list = [checkpoint]

    model.fit(x_data, y_data, epochs=500, batch_size=16, validation_split=0.15, shuffle=True, callbacks=callbacks_list,
              verbose=False)

    model.save('model.h5')

    train = False

# if not train:
#     model = createCDNet(True)
#     name = 'weights-improvement_3par-01-0.97.hdf5'
#     model.load_weights(name)
#     rand_test = np.random.randint(0, len(X_test), 1)
#     print(rand_test)
#
#
#     pred = model.predict(np.expand_dims(X_test[rand_test[0]], axis=0))
#
#     print(pred)
#
#     print(Y_test[rand_test])
#
#
#
#
#     plotting = []
#     statistics = []
#
#     for n, value in enumerate(pred[0]):
#         plotting.append(value - Y_test[0][n])
#         print(n, value - Y_test[0][n])
#
#
#     plt.plot(range(0, len(plotting)), plotting)
#     plt.xticks(range(0, len(plotting)), ['Alpha helix', 'Isolated beta-bridge', 'Strand', '3-10 helix', 'Pi helix', 'Turn', 'Bend', 'None'], rotation=45)
#     plt.show()


if not train:
    model = createCDNet(True, y_data.shape[1])
    name = 'weights-improvement-47-0.68.hdf5'
    model.load_weights(name)

    rand_test = np.random.randint(0, len(X_test), 1)
    print(rand_test)

    for m, test_data in enumerate(x_data):
        pred = model.predict(np.expand_dims(test_data, axis=0))

        #print(pred)

        #print(y_data[m])

        plotting = []
        statistics = []

        for n, value in enumerate(pred[0]):
            plotting.append(value - Y_test[0][n])
            # print(n, value - Y_test[0][n])

        plt.plot(range(0, len(plotting)), plotting)
        plt.xticks(range(0, len(plotting)),
                   ['Alpha helix', 'Isolated beta-bridge', 'Strand', 'Turn', 'Bend', 'None', '3-10 helix + Pi helix'],
                   rotation=45)
plt.show()
