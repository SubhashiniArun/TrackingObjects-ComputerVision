"""
Adapted from "The Implementation of Optical Flow in Neural Networks", Nicole Ku'ulei-lani Flett http://nrs.harvard.edu/urn-3:HUL.InstRepos:39011510
"""

import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


def train():
    image_data = np.loadtxt("train-data.csv", delimiter="\n", dtype=str)
    for i in range(0, len(image_data)):
        if (i + 1) % 3 != 0:
            image_data[i] = image_data[i].replace(',', '')
            image_data[i] = image_data[i].replace('\\', '/')

    inputs = []
    outputs = []

    x = 0
    while x < len(image_data) - 4:
        img1 = Image.open(image_data[x]).convert(mode='L')
        img2 = Image.open(image_data[x + 1]).convert(mode='L')
        vis = np.concatenate((img1, img2), axis=0)
        vis.resize((28, 28))
        inputs.append(vis)
        label = [float(i) for i in image_data[x + 2].split(',') if i.isdigit()]
        outputs.append(label)
        x += 3

    inputs = np.array(inputs, dtype="float") / 255.0
    outputs = np.array(outputs)

    (trainX, testX, trainY, testY) = train_test_split(
        inputs, outputs, test_size=0.25)

    trainX = np.reshape(trainX, (trainX.shape[0], 28, 28, -1))
    testX = np.reshape(testX, (testX.shape[0], 28, 28, -1))

    bs = 32  # default batch size
    model = Sequential()
    model.add(Conv2D(32, (5, 5), data_format='channels_last',
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])

    # reshape trainY
    temp = []

    for i in range(trainY.shape[0]):
        temp.append(np.array([trainY[i][0], trainY[i][1],
                              trainY[i][2], trainY[i][3], 0, 0]))

    trainY = np.array(temp)
    # reshape ends

    history = model.fit(trainX, trainY, epochs=100, batch_size=bs, verbose=1)

    pickle.dump(history.history, open('save.p', 'wb'))

    score_train = model.evaluate(trainX, trainY, batch_size=bs, verbose=1)

    # reshape testY
    temp = []

    for i in range(testY.shape[0]):
        temp.append(
            np.array([testY[i][0], testY[i][1], testY[i][2], testY[i][3], 0, 0]))

    testY = np.array(temp)
    # reshape ends

    score_test = model.evaluate(testX, testY, batch_size=bs, verbose=1)

    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights("model.h5")

    return history



if __name__ == '__main__':
    train()
