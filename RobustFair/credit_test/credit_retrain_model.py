import numpy
from tensorflow import keras
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_retrain import get_retrain_data


def retrain_classifier_model(x_train, y_train, epochs, batch_sizes, model_suffix):
    "重训练回归模型"
    model_input = keras.Input(shape=(21,), name="input")
    layer1 = layers.Dense(21, activation="relu")(model_input)
    layer2 = layers.Dense(21, activation="relu")(layer1)
    layer3 = layers.Dense(10, activation="relu")(layer2)
    layer4 = layers.Dense(10, activation="relu")(layer3)
    layer5 = layers.Dense(2, activation="relu")(layer4)
    layer6 = layers.Dense(2, activation="relu")(layer5)
    model_output = layers.Dense(2, activation="softmax")(layer6)
    model = keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mse', 'acc'])
    file_path = "../dataset/credit/retrain/{}.h5".format(model_suffix)
    C_point = ModelCheckpoint(file_path, monitor='mse', save_best_only=True)
    model.fit(x=x_train, y=y_train, shuffle=True, epochs=epochs, batch_size=batch_sizes, verbose=0, callbacks=C_point)


Epoch = 50
Batch_size = 128
Percentage = 0.10
Retrain_size = 10
if __name__ == "__main__":
    train_file = "../dataset/credit/data/train.npz.npy"
    for i in range(Retrain_size):
        # accurate fairness
        for g in ["RF"]:
            test_file1 = "../dataset/credit/test/BL_{}_{}_generation.npz.npy".format(i, g)
            retrain_x, retrain_y = get_retrain_data(train_file, test_file1, Percentage)
            retrain_y = to_categorical(retrain_y, num_classes=2)
            retrain_name = "{}_{}".format(g, i)
            retrain_classifier_model(retrain_x, retrain_y, Epoch, Batch_size, retrain_name)
