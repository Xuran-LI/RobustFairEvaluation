import numpy
import pandas


def combine_base_model_attack_evaluations(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    # accurate fairness
    for method in ["RF"]:
        file = "../dataset/{}/result/BL_{}_generation_result.xlsx".format(D_tag, method)
        result_data.append(["methods"] + pandas.read_excel(file).columns.values.reshape(1, -1).tolist()[0])

        eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
        result_data.append([method] + eval_data)
    return numpy.squeeze(numpy.array(result_data))


def combine_retrain_model_evaluations(D_tag):
    """
    获取重训练模型的评估结果
    :param D_tag:
    :return:
    """
    R_data = []
    # BL model
    file0 = "../dataset/{}/result/BL_model_evaluation.xlsx".format(D_tag)
    eval_data0 = pandas.read_excel(file0)
    R_data.append(["Model"] + eval_data0.columns.values.reshape(1, -1).tolist()[0])
    R_data.append(["BL"] + eval_data0.values[0, :].reshape(1, -1).tolist()[0])
    # retrain model
    for method in ["RF"]:
        file = "../dataset/{}/result/Re_{}_evaluation.xlsx".format(D_tag, method)
        eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
        R_data.append([method] + eval_data)

    return numpy.squeeze(numpy.array(R_data))


def combine_multi_seeds_attack_evaluations(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    seeds_num = [100, 200, 300, 400, 500]
    for seed in seeds_num:
        if seed == 200:
            file = "../dataset/{}/result/BL_RF_generation_result.xlsx".format(D_tag)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(seed)] + eval_data)
        else:
            file = "../dataset/{}/result/Seeds_{}_BL_RF_generation_result.xlsx".format(D_tag, seed)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(seed)] + eval_data)

    return numpy.squeeze(numpy.array(result_data))


def combine_multi_iteration_local_evaluations(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    iter_num = [5, 10, 15, 20, 25, 30]
    for iter in iter_num:
        if iter == 5:
            file = "../dataset/{}/result/BL_RF_generation_result.xlsx".format(D_tag)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(iter)] + eval_data)
        else:
            file = "../dataset/{}/result/Iterations_{}_BL_RF_generation_result.xlsx".format(D_tag, iter)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(iter)] + eval_data)

    return numpy.squeeze(numpy.array(result_data))


def combine_multi_iteration_global_evaluations(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    iter_num = [5, 10, 15, 20, 25, 30]
    for iter in iter_num:
        if iter == 5:
            file = "../dataset/{}/result/BL_RF_generation_result.xlsx".format(D_tag)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(iter)] + eval_data)
        else:
            file = "../dataset/{}/result/G_Iterations_{}_BL_RF_generation_result.xlsx".format(D_tag, iter)
            eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
            result_data.append([str(iter)] + eval_data)

    return numpy.squeeze(numpy.array(result_data))
