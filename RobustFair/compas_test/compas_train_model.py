import os
import numpy
import pandas
from tensorflow import keras
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_generate import data_augmentation_compas


def get_classifier_model(x_train, y_train, epochs, batch_sizes, model_name):
    "训练3层深度回归模型"
    model_input = keras.Input(shape=(11,), name="input")
    layer1 = layers.Dense(32, activation="relu")(model_input)
    layer2 = layers.Dense(64, activation="relu")(layer1)
    layer3 = layers.Dense(128, activation="relu")(layer2)
    layer4 = layers.Dense(64, activation="relu")(layer3)
    layer5 = layers.Dense(32, activation="relu")(layer4)
    layer6 = layers.Dense(2, activation="relu")(layer5)
    model_output = layers.Dense(2, activation="softmax")(layer6)
    model = keras.Model(inputs=model_input, outputs=model_output)
    model.summary()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=["mse", 'acc'])
    file_path = "../dataset/compas/model/{}.h5".format(model_name)
    C_point = ModelCheckpoint(file_path, monitor='mse', save_best_only=True)
    model.fit(x=x_train, y=y_train, shuffle=True, epochs=epochs, batch_size=batch_sizes, verbose=1, callbacks=C_point)


epoch = 50
batch_size = 128
if __name__ == "__main__":
    if not os.path.exists("../dataset/compas/data/train.npz.npy"):
        compas_data = pandas.read_csv("../dataset/compas/data/train.txt", header=None).values
        numpy.save("../dataset/compas/data/train.npz.npy", compas_data)

    if not os.path.exists("../dataset/compas/data/test.npz.npy"):
        compas_data = pandas.read_csv("../dataset/compas/data/test.txt", header=None).values
        numpy.save("../dataset/compas/data/test.npz.npy", compas_data)

    if not os.path.exists("../dataset/compas/data/multiple_train.npz.npy"):
        data1 = numpy.load("../dataset/compas/data/train.npz.npy")
        aug = data_augmentation_compas(data1, [0, 1, 2]),
        numpy.save("../dataset/compas/data/multiple_train.npz.npy", aug)

    if not os.path.exists("../dataset/compas/data/multiple_test.npz.npy"):
        test_data = numpy.load("../dataset/compas/data/test.npz.npy")
        aug = data_augmentation_compas(test_data, [0, 1, 2]),
        numpy.save("../dataset/compas/data/multiple_test.npz.npy", aug)

    "compas 预测模型"
    "base line"
    train_data = numpy.load("../dataset/compas/data/train.npz.npy")
    train_x, train_y = numpy.split(train_data, [11, ], axis=1)
    train_y = to_categorical(train_y, num_classes=2)
    get_classifier_model(train_x, train_y, epoch, batch_size, "BL")
