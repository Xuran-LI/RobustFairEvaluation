import numpy

from utils.utils_evaluate import calculate_cos


def get_retrain_data(train_file, cluster_file, generation_file, cos_threshold):
    """
    将生成样本作为重训练样本
    :return:
    """
    train_data = numpy.load(train_file)
    retrain_x, retrain_y = numpy.split(train_data, [-1, ], axis=1)

    cluster_data = numpy.load(cluster_file)
    # 生成样本
    generate_data = numpy.load(generation_file)
    cos_result, select_cond = calculate_cos(cluster_data, generate_data, cos_threshold)
    select_x, select_y = numpy.split(generate_data[select_cond], [-1, ], axis=1)

    retrain_x = numpy.concatenate((retrain_x, select_x), axis=0)
    retrain_y = numpy.concatenate((retrain_y, select_y), axis=0)

    return retrain_x, retrain_y
