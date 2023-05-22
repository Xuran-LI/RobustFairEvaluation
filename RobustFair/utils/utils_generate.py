from random import uniform
from sklearn import cluster
import numpy
import tensorflow
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.utils.np_utils import to_categorical


# "计算损失函数梯度"
def compute_loss_grad(x, label, model, loss_func=mean_squared_error):
    """计算模型损失函数相对于输入features的导数 dloss(f(x),y)/d(x)"""
    x = tensorflow.constant(x, dtype=tensorflow.float32)
    label = tensorflow.constant([label], dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(label, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


def compute_loss_grad_MP(x_list, label, model, max_id, loss_func=mean_squared_error):
    """计算相似样本距离函数对非保护属性X的最大导数 D(f(x,a'),y)"""
    label = tensorflow.constant([label], dtype=tensorflow.float32)
    gradients = []
    for similar in x_list:
        similar = tensorflow.constant(similar.reshape(1, -1), dtype=tensorflow.float32)
        with tensorflow.GradientTape() as tape:
            tape.watch(similar)
            D_distance = loss_func(label, model(similar))
        gradient = tape.gradient(D_distance, similar)
        gradients.append(gradient[0].numpy())
    return gradients[max_id]


def far_similar_C(pre_list, pre):
    """返回相似样本集中预测结果与真实标记差别最大的索引，分类任务"""
    initial_id = 0
    initial_gap = numpy.sum(abs(pre_list[0] - pre[0]))
    for i in range(pre_list.shape[0]):
        if numpy.sum(abs(pre_list[i] - pre[0])) > initial_gap:
            initial_id = i
            initial_gap = abs(pre_list[i] - pre[0])[0]
    return initial_id


# 聚类生成种子
def clustering(data_x, data_y, c_num):
    # standard KMeans algorithm
    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data_x)
    return [data_x[y_pred == n] for n in range(c_num)], [data_y[y_pred == n] for n in range(c_num)]


def random_pick(probability):
    # randomly pick an element from a probability distribution
    random_number = numpy.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(data_x, data_y, data_len, c_num):
    pick_probability = [len(data_x[i]) / data_len for i in range(c_num)]
    cluster_i = random_pick(pick_probability)
    seed_x = data_x[cluster_i]
    seed_y = data_y[cluster_i]
    index = numpy.random.randint(0, len(seed_x))
    return seed_x[index], seed_y[index]


def get_global_seeds(test_file, cluster_num, sample_num):
    """
    获取全局搜索种子
    :return:
    """
    seeds = []
    test_data = numpy.load(test_file)
    test_x, test_y = numpy.split(test_data, [-1, ], axis=1)
    cluster_x, cluster_y = clustering(test_x, test_y, cluster_num)
    for i in range(sample_num):
        s_x, s_y = get_seed(cluster_x, cluster_y, test_x.shape[0], cluster_num)
        seeds.append(numpy.concatenate((s_x, s_y), axis=0))
    return numpy.squeeze(numpy.array(seeds))


# 基于泰勒公式计算生成样本的标记
def multiple_matrix(m1, m2):
    result = 0
    for i in range(m1.shape[1]):
        result += m1[0, i] * m2[i, 0]
    return result


def compute_label_Taylor_C(x1, label1, gradient1, perturbation, x2, model, loss_func=mean_squared_error):
    "根据泰勒公式，近似计算生成样本的真实标记，分类任务"
    label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
    x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
    x2 = tensorflow.constant(x2, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x1)
        pre2 = model(x2)
        loss1 = loss_func(label1, model(x1)).numpy()
    # loss2=loss1+loss'*perturbation
    loss2 = multiple_matrix(perturbation.reshape(1, -1), gradient1.reshape(-1, 1)) + loss1
    # 判断 [1,0]与【0，1】的损失与loss2的接近程度
    loss2_0 = loss_func(to_categorical(0, num_classes=2), pre2)
    loss2_1 = loss_func(to_categorical(1, num_classes=2), pre2)
    if abs(loss2_0 - loss2) <= abs(loss2_1 - loss2):
        return [0]
    else:
        return [1]


#  对扰动顺序进行排序
def sort_perturbation_direction(direction, protected):
    """
    对非保护属性梯度，按从小到大的顺序排序
    :return:
    """
    sort_result = []
    for i in range(direction.shape[1]):
        min_id = i
        min_data = abs(direction[0, i])
        for j in range(direction.shape[1]):
            if abs(direction[0, j]) < min_data:
                min_id = j
                min_data = direction[0, j]
            j += 1
        #  设置最小值位置为无穷大
        direction[0, min_id] = 1000
        if min_id not in protected and direction[0, min_id] != 0:
            sort_result.append(min_id)
    return sort_result


# 数据增强
def data_augmentation_adult(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
            for a_1 in [uniform(0, 2), uniform(2, 4)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (4)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    # data_list.append(aug_data)
                    # 生成测试数据集相似样本时，去除真实标记
                    data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_adult_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
            for a_1 in [uniform(0, 2), uniform(2, 4)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (4)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_credit(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [0, 1]:
            for a_1 in [uniform(19, 35), uniform(35, 55), uniform(55, 75)]:
                aug_data = data[i].tolist()
                aug_data[protected_index[0]] = a_0
                aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
                # data_list.append(aug_data)
                # 生成测试数据集相似样本时，去除真实标记
                data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_credit_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [0, 1]:
            for a_1 in [uniform(19, 35), uniform(35, 55), uniform(55, 75)]:
                aug_data = data[i].tolist()
                aug_data[protected_index[0]] = a_0
                aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_compas(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(18, 30), uniform(30, 60), uniform(60, 96)]:
            for a_1 in [uniform(0, 2.5), uniform(2.5, 5)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (5)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    # data_list.append(aug_data)
                    # 生成测试数据集相似样本时，去除真实标记
                    data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_compas_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(18, 30), uniform(30, 60), uniform(60, 96)]:
            for a_1 in [uniform(0, 2.5), uniform(2.5, 5)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (5)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_bank(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_1 in [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]:
            aug_data = data[i].tolist()
            aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
            # data_list.append(aug_data)
            # 生成测试数据集相似样本时，去除真实标记
            data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_bank_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_1 in [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]:
            aug_data = data[i].tolist()
            aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
            data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


# gradient normalization during local search
def normal_prob(grad1, grad2, protected_attribs, epsilon):
    gradient = numpy.zeros_like(grad1)
    grad1 = numpy.abs(grad1)
    grad2 = numpy.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attribs:
            gradient[i] = 0.0
    gradient_sum = numpy.sum(gradient)
    probability = gradient / gradient_sum
    return probability
