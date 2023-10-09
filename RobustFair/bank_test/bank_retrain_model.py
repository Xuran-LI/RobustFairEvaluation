from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_retrain import get_retrain_data


def retrain_classifier_model(model_file, x_train, y_train, epochs, batch_sizes, model_suffix):
    """重训练回归模型"""
    model_input = keras.Input(shape=(16,), name="input")
    layer1 = layers.Dense(32, activation="relu")(model_input)
    layer2 = layers.Dense(64, activation="relu")(layer1)
    layer3 = layers.Dense(128, activation="relu")(layer2)
    layer4 = layers.Dense(64, activation="relu")(layer3)
    layer5 = layers.Dense(32, activation="relu")(layer4)
    layer6 = layers.Dense(2, activation="relu")(layer5)
    model_output = layers.Dense(2, activation="softmax")(layer6)
    model = keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mse', "acc"])
    file_path = "../dataset/bank/retrain/{}.h5".format(model_suffix)
    C_point = ModelCheckpoint(file_path, monitor='mse', save_best_only=True)
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_sizes, verbose=0, callbacks=C_point)


Epoch = 100
Batch_size = 512
Retrain_size = 1
if __name__ == "__main__":
    train_file = "../dataset/bank/data/train.npz.npy"
    cluster_file = "../dataset/bank/test/retrain_avg_clusters.npz.npy"
    BL_model_file = "../dataset/bank/model/BL.h5"
    for cos in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for i in range(Retrain_size):
            for m in ["RobustFair"]:
                generate_file = "../dataset/bank/test/Train_{}_{}_10_10.npz.npy".format(m, i)
                retrain_x, retrain_y = get_retrain_data(train_file, cluster_file, generate_file, cos)
                retrain_y = to_categorical(retrain_y, num_classes=2)
                retrain_name = "{}_{}_{}".format(cos, m, i)
                retrain_classifier_model(BL_model_file, retrain_x, retrain_y, Epoch, Batch_size, retrain_name)
