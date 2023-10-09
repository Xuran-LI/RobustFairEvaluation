import numpy
import pandas


def combine_model_attack_evaluations10(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    for method in ["RobustFair"]:
        E_file = "../dataset/{}/result/Test_{}_100_0.xlsx".format(D_tag, method)
        E_data = numpy.mean(pandas.read_excel(E_file).values, axis=0).tolist()
        result_data.append([method] + E_data)
    for method in ["RobustFair"]:
        E_file = "../dataset/{}/result/Test_{}_0_100.xlsx".format(D_tag, method)
        E_data = numpy.mean(pandas.read_excel(E_file).values, axis=0).tolist()
        result_data.append([method] + E_data)
    return numpy.squeeze(numpy.array(result_data))


def combine_model_attack_evaluations100(D_tag):
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    # robustness
    for method in ["RobustFair"]:
        E_file = "../dataset/{}/result/Test_{}_10_10.xlsx".format(D_tag, method)
        E_data = numpy.mean(pandas.read_excel(E_file).values, axis=0).tolist()
        result_data.append([method] + E_data)
    return numpy.squeeze(numpy.array(result_data))


def combine_retrain_model_evaluations(D_tag, cos):
    """
    获取重训练模型的评估结果
    :return:
    """
    R_data = []
    # BL model
    file0 = "../dataset/{}/result/Train_BL_evaluation.xlsx".format(D_tag)
    eval_data = pandas.read_excel(file0)
    R_data.append(["Model"] + eval_data.columns.values.reshape(1, -1).tolist()[0])
    R_data.append(["BL"] + eval_data.values[0, :].reshape(1, -1).tolist()[0])
    # retrain model
    for method in ["RobustFair"]:
        file = "../dataset/{}/result/Retrain_{}_{}_evaluation.xlsx".format(D_tag, cos, method)
        eval_data = numpy.mean(pandas.read_excel(file).values, axis=0).tolist()
        R_data.append([method] + eval_data)
    return numpy.squeeze(numpy.array(R_data))


def combine_model_attack_evaluations10_datasets():
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    # robustness
    for method in ["RobustFair"]:
        method_data = []
        for D_tag in ["adult", "bank", "compas", "credit"]:
            E_file = "dataset/{}/result/Test_{}_100_0.xlsx".format(D_tag, method)
            method_data.append(numpy.mean(pandas.read_excel(E_file).values, axis=0))
        E_data = numpy.mean(method_data, axis=0).tolist()
        result_data.append([method] + E_data)

    for method in ["RobustFair"]:
        method_data = []
        for D_tag in ["adult", "bank", "compas", "credit"]:
            E_file = "dataset/{}/result/Test_{}_0_100.xlsx".format(D_tag, method)
            a = numpy.mean(pandas.read_excel(E_file).values, axis=0)
            method_data.append(numpy.mean(pandas.read_excel(E_file).values, axis=0))
        E_data = numpy.mean(method_data, axis=0).tolist()
        result_data.append([method] + E_data)
    return numpy.squeeze(numpy.array(result_data))


def combine_model_attack_evaluations100_datasets():
    """
    获取BL模型的攻击评估结果
    :return:
    """
    result_data = []
    # robustness
    for method in ["RobustFair"]:

        method_data = []
        for D_tag in ["adult", "bank", "compas", "credit"]:
            E_file = "dataset/{}/result/Test_{}_10_10.xlsx".format(D_tag, method)
            method_data.append(numpy.mean(pandas.read_excel(E_file).values, axis=0))

        E_data = numpy.mean(method_data, axis=0).tolist()
        result_data.append([method] + E_data)
    return numpy.squeeze(numpy.array(result_data))


def combine_retrain_model_evaluations_datasets(cos):
    """
    获取重训练模型的评估结果
    :param D_tag:
    :return:
    """
    R_data = []
    # BL model
    method_data = []
    for D_tag in ["adult", "bank", "compas", "credit"]:
        file0 = "dataset/{}/result/Train_BL_evaluation.xlsx".format(D_tag)
        eval_data = pandas.read_excel(file0)
        method_data.append(numpy.mean(pandas.read_excel(file0).values, axis=0))

    R_data.append(["Model"] + eval_data.columns.values.reshape(1, -1).tolist()[0])
    R_data.append(["BL"] + numpy.mean(method_data, axis=0).tolist())
    # retrain model
    for method in ["RobustFair"]:
        method_data = []
        for D_tag in ["adult", "bank", "compas", "credit"]:
            file = "dataset/{}/result/Retrain_{}_{}_evaluation.xlsx".format(D_tag, cos, method)
            method_data.append(numpy.mean(pandas.read_excel(file).values, axis=0))

        eval_data = numpy.mean(method_data, axis=0).tolist()
        R_data.append([method] + eval_data)
    return numpy.squeeze(numpy.array(R_data))
