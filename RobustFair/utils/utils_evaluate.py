import math

import numpy
import xlsxwriter
from tensorflow.python.keras.saving.save import load_model
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data


def calculate_MSE(data1, data2):
    """
    计算均方误差
    :return:
    """
    MSE_distance = []
    for i in range(data1.shape[0]):
        num = 0
        for j in range(data1.shape[1]):
            num += ((data1[i, j] - data2[i, j]) * (data1[i, j] - data2[i, j]))
        distance = num / data1.shape[1]
        MSE_distance.append(distance)
    return numpy.array(MSE_distance).astype(float)


def check_dist(data, dist):
    """
    检查data中小于dist的结果
    :return:
    """
    result = []
    for i in range(data.shape[0]):
        if data[i] <= dist:
            result.append(True)
        else:
            result.append(False)
    return numpy.array(result).astype(bool)


def compare_data(data1, data2):
    """
    检查data中小于dist的结果
    :return:
    """
    result = []
    for i in range(data1.shape[0]):
        if data1[i] <= data2[i]:
            result.append(True)
        else:
            result.append(False)
    return numpy.array(result).astype(bool)


def check_performance(pre, label, dist):
    """
    计算模型的performance
    :return:
    """
    # 预测结果与真实标记MSE的平均值，方差
    MSE = calculate_MSE(pre, label)
    avg = numpy.mean(MSE)
    std = numpy.std(MSE)
    # 预测结果与真实标记MSE小于dist的比例
    acc_n = numpy.sum(check_dist(MSE, dist))
    acc_r = acc_n / label.shape[0]
    false_n = numpy.sum(~check_dist(MSE, dist))
    false_r = false_n / label.shape[0]
    return [acc_r, acc_n, false_r, false_n]


def check_IF(pre1, pre2, x1, x2, dist, K):
    """
    检查 individual fair
    模型预测结果是否满足 D(f(x1),f(x2))<=Kd(x1,x2)
    :return:
    """
    IF_cond = numpy.ones(pre1.shape[0])
    for i in range(len(pre2)):
        # D(f(x1),f(x2))<=Kd(x1,x2)+dist
        D_distance = calculate_MSE(pre1, pre2[i])
        Kd_distance = K * calculate_MSE(x1, x2[i])
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_num = numpy.sum(IF_cond)
    IF_rate = IF_num / x1.shape[0]
    IB_num = numpy.sum(~IF_cond)
    IB_rate = IB_num / x1.shape[0]
    return [IF_rate, IF_num, IB_rate, IB_num]


def check_item_IF(pre1, pre2, x1, x2, dist, K):
    """
    检查样本与相似样本的预测结果是否 individual fair
    D(f(x1),f(x2))<=Kd(x1,x2)
    :return:
    """
    for i in range(len(pre2)):
        D_distance = calculate_MSE(pre1.reshape(1, 1), pre2[i].reshape(1, 1))
        Kd_distance = K * calculate_MSE(x1, x2[i].reshape(1, -1))
        IF_result = check_dist(D_distance - Kd_distance, dist)
        if IF_result[0]:
            pass
        else:
            return False
    return True


def check_AF(label, pre1, pre2, x1, x2, dist, K):
    """
    检查待测样本的准确公平性
    :param label: 待测样本标签
    :param pre1: 待测样本预测结果
    :param pre2: 待测样本的相似样本预测结果集合
    :param x1: 待测样本
    :param x2: 待测样本相似样本集合
    :param dist: 准确公平性超参数1
    :param K: 准确公平性超参数2
    :return: 待测样本准确且公平率，错误或歧视率
    """
    MSE = calculate_MSE(pre1, label)
    AF_cond1 = check_dist(MSE, dist)
    AF_cond2 = numpy.ones(AF_cond1.shape)
    for i in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
        D_distance = calculate_MSE(label, pre2[i])
        Kd_distance = K * calculate_MSE(x1, x2[i])
        AF_cond2 = numpy.logical_and(AF_cond2, check_dist(D_distance - Kd_distance, dist))

    # 计算 T&F 与 T|B
    AF_cond = numpy.logical_and(AF_cond1, AF_cond2)
    FB_cond = ~AF_cond
    AF = numpy.sum(AF_cond)
    FB = numpy.sum(FB_cond)
    AFR = numpy.sum(AF_cond) / x1.shape[0]
    FBR = numpy.sum(FB_cond) / x1.shape[0]

    return [AFR, AF, FBR, FB]


def check_item_AF(label, pre1, pre2, x1, x2, dist, K):
    """
    检查数据集的 accurate fairness
    """
    MSE = calculate_MSE(pre1.reshape(1, 1), label.reshape(1, 1))
    AF_result = check_dist(MSE, dist)
    for i in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
        D_distance = calculate_MSE(label.reshape(1, 1), pre2[i].reshape(1, 1))
        Kd_distance = K * calculate_MSE(x1, x2[i].reshape(1, -1))
        AF_result = numpy.logical_and(AF_result, check_dist(D_distance - Kd_distance, dist))
        if AF_result[0]:
            pass
        else:
            return False
    return True


def check_dataset_confusion(label, pre1, pre2, x1, x2, dist, K):
    """
    计算 fairness confusion
    """
    MSE = calculate_MSE(pre1, label)
    True_cond = check_dist(MSE, dist)
    AF_cond = numpy.ones(True_cond.shape)
    for j in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+dist
        D_distance = calculate_MSE(label, pre2[j])
        Kd_distance = K * calculate_MSE(x1, x2[j])
        AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_cond = numpy.ones(True_cond.shape)
    for i in range(len(pre2)):
        # D(f(x),f(similar_x))<=Kd(x,similar_x)+dist
        D_distance = calculate_MSE(pre1, pre2[i])
        Kd_distance = K * calculate_MSE(x1, x2[i])
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 TF TB FF FB
    TF_cond = numpy.logical_and(True_cond, IF_cond)
    TB_cond = numpy.logical_and(True_cond, ~IF_cond)
    FF_cond = numpy.logical_and(~True_cond, IF_cond)
    FB_cond = numpy.logical_and(~True_cond, ~IF_cond)

    TF = numpy.sum(TF_cond)
    TB = numpy.sum(TB_cond)
    FF = numpy.sum(FF_cond)
    FB = numpy.sum(FB_cond)

    TFR = numpy.sum(TF_cond) / x1.shape[0]
    TBR = numpy.sum(TB_cond) / x1.shape[0]
    FFR = numpy.sum(FF_cond) / x1.shape[0]
    FBR = numpy.sum(FB_cond) / x1.shape[0]

    if (TFR + FFR) == 0:
        F_recall = 0
    else:
        F_recall = TFR / (TFR + FFR)

    if (TFR + TBR) == 0:
        F_precision = 0
    else:
        F_precision = TFR / (TFR + TBR)

    if (F_recall + F_precision) == 0:
        F_F1 = 0
    else:
        F_F1 = (2 * F_recall * F_precision) / (F_recall + F_precision)

    return [TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1, TF, TB, FF, FB, TF + TB + FF + FB]


def get_AF_condition(label, pre1, pre2, x1, x2, dist, K):
    """
    计算 fairness confusion
    """
    MSE = calculate_MSE(pre1, label)
    True_cond = check_dist(MSE, dist)
    AF_cond = numpy.ones(True_cond.shape)
    for j in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+dist
        D_distance = calculate_MSE(label, pre2[j])
        Kd_distance = K * calculate_MSE(x1, x2[j])
        AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))

    IF_cond = numpy.ones(True_cond.shape)
    for i in range(len(pre2)):
        # D(f(x),f(similar_x))<=Kd(x,similar_x)+dist
        D_distance = calculate_MSE(pre1, pre2[i])
        Kd_distance = K * calculate_MSE(x1, x2[i])
        IF_cond = numpy.logical_and(IF_cond, check_dist(D_distance - Kd_distance, dist))

    # 计算 TF TB FF FB
    TF_cond = numpy.logical_and(True_cond, IF_cond)
    TB_cond = numpy.logical_and(True_cond, ~IF_cond)
    FF_cond = numpy.logical_and(~True_cond, IF_cond)
    FB_cond = numpy.logical_and(~True_cond, ~IF_cond)

    return [TF_cond, TB_cond, FF_cond, FB_cond]


# 对模型进行评估
def regression_evaluation(file1, file2, file3, dist=0.001, K=0):
    """
    根据模型获取结果，回归模型
    :return:
    """
    model = load_model(file1)
    data1 = numpy.load(file2)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre1 = model.predict(x1)
    data2 = numpy.squeeze(numpy.load(file3))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(model.predict(data2[:, j, :]))
    ACC_result = check_performance(pre1, y1, dist)
    IF_result = check_IF(pre1, pre2, x1, x2, dist, K)
    AF_result = check_AF(y1, pre1, pre2, x1, x2, dist, K)
    Fair_confusion = check_dataset_confusion(y1, pre1, pre2, x1, x2, dist, K)
    return ACC_result + IF_result + AF_result + Fair_confusion


def check_regression(model_files, file1, file2, eval_file, dist):
    """
    评估分类模型
    :return:
    """
    header_name = ["avg", "std", "acc R", "false R", "acc N", "false N", "SUM",
                   "IFR", "IBR", "IFN", "IBN", "SUM",
                   "A&F R", "F|B R", "A&F", "F|B", "SUM",
                   "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM"]
    evaluation_data = []
    for f in model_files:
        result_evaluation = regression_evaluation(f, file1, file2, dist)
        evaluation_data.append(result_evaluation)
        print(result_evaluation)

    workbook_name = xlsxwriter.Workbook(eval_file)
    worksheet = workbook_name.add_worksheet("Generation Details")
    write_worksheet_header(header_name, worksheet)
    write_worksheet_2d_data(evaluation_data, worksheet)
    workbook_name.close()


def classification_evaluation1(file1, file2, file3, S_file, dist=0, K=0):
    """
    根据模型获取结果，分类模型
    :return:
    """
    model = load_model(file1)
    data1 = numpy.load(file2)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    data2 = numpy.squeeze(numpy.load(file3))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))
    ACC_result = check_performance(pre, y1, dist)
    IF_result = check_IF(pre, pre2, x1, x2, dist, K)
    AF_result = check_AF(y1, pre, pre2, x1, x2, dist, K)
    Fair_confusion = check_dataset_confusion(y1, pre, pre2, x1, x2, dist, K)

    cluster_data = numpy.load(S_file)
    cos_result, select_cond = calculate_cos(cluster_data, data1, 1.0)
    return ACC_result + IF_result + AF_result + Fair_confusion + [numpy.mean(cos_result)]


def calculate_cos(cluster_data, generate_data, cos_threshold):
    """
    计算生成样本与聚类中心的余弦相似度，并选择大于cos_threshold的位置
    :return:
    """
    condition = []
    cos_result = []
    for i in range(generate_data.shape[0]):
        cos_max = 0
        for j in range(cluster_data.shape[0]):
            if cos(cluster_data[j], generate_data[i]) > cos_max:
                cos_max = cos(cluster_data[j], generate_data[i])
        cos_result.append(cos_max)
        if cos_max > cos_threshold:
            condition.append(True)
        else:
            condition.append(False)

    return numpy.array(cos_result), condition


def classification_evaluation2(file1, file2, file3, dist=0, K=0):
    """
    根据模型获取结果，分类模型
    :return:
    """
    model = load_model(file1)
    data1 = numpy.load(file2)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    data2 = numpy.squeeze(numpy.load(file3))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))
    ACC_result = check_performance(pre, y1, dist)
    IF_result = check_IF(pre, pre2, x1, x2, dist, K)
    AF_result = check_AF(y1, pre, pre2, x1, x2, dist, K)
    Fair_confusion = check_dataset_confusion(y1, pre, pre2, x1, x2, dist, K)

    return ACC_result + IF_result + AF_result + Fair_confusion


def cos(a, b):  #
    dot = 0
    mod_a = 0
    mod_b = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
        mod_a += a[i] * a[i]
        mod_b += b[i] * b[i]

    return dot / (math.sqrt(mod_a) * math.sqrt(mod_b))


def check_classification1(model_files, file1, file2, S_file, eval_file):
    """
    评估分类模型
    :return:
    """
    header_name = ["Acc R", "acc N", "False R", "False N",
                   "IF R", "IF N", "IB R", "IB N",
                   "A&F R", "A&F", "F|B R", "F|B",
                   "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM",
                   "cos"]
    evaluation_data = []
    for f in model_files:
        result_evaluation = classification_evaluation1(f, file1, file2, S_file)
        evaluation_data.append(result_evaluation)
        print(result_evaluation)

    workbook_name = xlsxwriter.Workbook(eval_file)
    worksheet = workbook_name.add_worksheet("Generation Details")
    write_worksheet_header(header_name, worksheet)
    write_worksheet_2d_data(evaluation_data, worksheet)
    workbook_name.close()


def check_classification2(model_files, file1, file2, eval_file):
    """
    评估分类模型
    :return:
    """
    header_name = ["Acc R", "acc N", "False R", "False N",
                   "IF R", "IF N", "IB R", "IB N",
                   "A&F R", "A&F", "F|B R", "F|B",
                   "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM"]
    evaluation_data = []
    for f in model_files:
        result_evaluation = classification_evaluation2(f, file1, file2)
        evaluation_data.append(result_evaluation)
        print(result_evaluation)

    workbook_name = xlsxwriter.Workbook(eval_file)
    worksheet = workbook_name.add_worksheet("Generation Details")
    write_worksheet_header(header_name, worksheet)
    write_worksheet_2d_data(evaluation_data, worksheet)
    workbook_name.close()


