import numpy


def get_retrain_data(train_file, generation_file, percentage):
    """
    将生成样本作为重训练样本
    :return:
    """
    # 原始训练集
    data = numpy.load(train_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)

    # 生成集合
    data1 = numpy.load(generation_file)
    data1_x, data1_y = numpy.split(data1, [-1, ], axis=1)

    # 随机选择生成集合内数据
    data_index = numpy.arange(data1_x.shape[0])
    numpy.random.shuffle(data_index)
    retrain_num = round(percentage * data1_x.shape[0])
    select_index = data_index[:retrain_num]
    select_x = data1_x[select_index]
    select_y = data1_y[select_index]

    # 生成重训练集合
    retrain_x = numpy.concatenate((data_x, select_x), axis=0)
    retrain_y = numpy.concatenate((data_y, select_y), axis=0)

    return retrain_x, retrain_y
