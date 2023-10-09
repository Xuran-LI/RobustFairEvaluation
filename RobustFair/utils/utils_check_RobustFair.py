import keras
import numpy
import xlsxwriter
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.utils_draw import draw_lines_loss, draw_lines_num
from utils.utils_evaluate import calculate_MSE, check_dist
from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad_MP, approximate_by_calculus_Max, \
    compute_loss_grad, sort_perturbation_direction, data_augmentation_compas_item, data_augmentation_credit_item, \
    data_augmentation_bank_item, get_AF_seed, get_AF_seeds, get_search_seeds
from utils.utils_input_output import write_worksheet_2d_data


def check_false_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] != 0:
                        direction[0, n] = sign1[n]
                    else:
                        direction[0, n] = -1 * sign2[n]
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)

            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_true_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] != 0:
                        direction[0, n] = -1 * sign1[n]
                    else:
                        direction[0, n] = sign2[n]

            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_false_fair_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] == sign2[n]:
                    direction[0, n] = sign1[n]

            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_false_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = sign1[n]
                else:
                    direction[0, n] = -1 * sign2[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:  # 根据梯度方向进行局部搜索
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_true_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = -1 * sign1[n]
                else:
                    direction[0, n] = sign2[n]
                    # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_false_fair_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

        pert_x = []
        pert_y = []
        pert_loss = []

        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] == sign2[n]:
                direction[0, n] = sign1[n]

        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            #  扰动所选择属性
            perturbation = extent * perturbation_direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            x_i = generated_x
            y_i = pert_l
            pert_x.append(x_i)
            pert_y.append(y_i)
            pert_loss.append(l_p)

        generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
        loss_perturbation.append(pert_loss)

    return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))


def check_global_generation(model, global_seeds, dataset, protected, search_times, extent, check_tag):
    """
    全局搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    if check_tag == "FB_G":
        R_item, R_loss = check_false_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    elif check_tag == "TB_G":
        R_item, R_loss = check_true_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    elif check_tag == "FF_G":
        R_item, R_loss = check_false_fair_global(model, global_seeds, dataset, protected, search_times, extent)

    return R_item, R_loss


def check_local_generation(model, local_seeds, dataset, protected, search_times, extent, check_tag):
    """
    局部搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    if check_tag == "FB_L":
        R_item, R_loss = check_false_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    elif check_tag == "TB_L":
        R_item, R_loss = check_true_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    elif check_tag == "FF_L":
        R_item, R_loss = check_false_fair_local(model, local_seeds, dataset, protected, search_times, extent)

    return R_item, R_loss


def check_FCD_loss(model_file, test_file, similar_file, dataset, protected, t1, t2, extent, AF_tag, check_tag):
    """
    检测进行 true bias， false bias， false fair 扰动时， 损失函数的变化情况
    :return:
    """
    model = keras.models.load_model(model_file)
    search_seeds = get_AF_seed(model_file, test_file, similar_file, AF_tag)
    if check_tag in {"TB_G", "FB_G", "FF_G"}:
        item, loss = check_global_generation(model, search_seeds, dataset, protected, t1, extent, check_tag)
    else:
        item, loss = check_local_generation(model, search_seeds, dataset, protected, t2, extent, check_tag)

    return numpy.array(item), numpy.array(loss)


def check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, s_t, extent, AF_tag, C_tag, pic_name, dist=0, K=0):
    """
    AF 测试样本生成
    :return:
    """
    while True:
        RF_R, RF_L = check_FCD_loss(M_file, T_file1, T_file2, D_tag, P_attr, s_t, s_t, extent, AF_tag, C_tag)
        if numpy.any(numpy.subtract(RF_L[:, 0], RF_L[:, 1]) > 0.001):
            print("data:{},search:{}".format(AF_tag, pic_name))
            break

    model = load_model(M_file)
    x1, y1 = numpy.split(RF_R, [-1, ], axis=1)
    pre1 = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    MSE = calculate_MSE(pre1, y1)
    Acc_Cond = check_dist(MSE, dist)

    # 生成相似样本
    if D_tag == "adult":
        similar_items = data_augmentation_adult_item(x1, P_attr)
    elif D_tag == "compas":
        similar_items = data_augmentation_compas_item(x1, P_attr)
    elif D_tag == "credit":
        similar_items = data_augmentation_credit_item(x1, P_attr)
    elif D_tag == "bank":
        similar_items = data_augmentation_bank_item(x1, P_attr)

    x2 = []
    pre2 = []
    for j in range(similar_items.shape[1]):
        x2.append(similar_items[:, j, :])
        pre2.append(numpy.argmax(model.predict(similar_items[:, j, :]), axis=1).reshape(-1, 1))

    AF_cond = numpy.ones(Acc_Cond.shape)

    for h in range(len(pre2)):
        # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
        D_distance = calculate_MSE(y1, pre2[h])
        Kd_distance = K * calculate_MSE(x1, x2[h])
        AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))

    TF_num = compute_cumulative(numpy.logical_and(Acc_Cond, AF_cond))
    TB_num = compute_cumulative(numpy.logical_and(Acc_Cond, ~AF_cond))
    FF_num = compute_cumulative(numpy.logical_and(~Acc_Cond, AF_cond))
    FB_num = compute_cumulative(numpy.logical_and(~Acc_Cond, ~AF_cond))

    x = [a for a in range(RF_R.shape[0])]
    y = [TF_num, TB_num, FF_num, FB_num]
    names = ["TF", "TB", "FF", "FB"]
    s_eval = "../dataset/{}/result/Pic_Max_{}_D_{}_search_num.pdf".format(D_tag, AF_tag, pic_name)
    draw_lines_num(x, y, names, x_label="Iterations", y_label="Number", P_title=pic_name, output_file=s_eval)

    loss_change = RF_L[:, 0]
    loss_change_similar = RF_L[:, 1]
    y = [loss_change, loss_change_similar]
    x = [a for a in range(RF_L.shape[0])]
    names = ["Perturbed Individual", "Perturbed Similar Individual"]
    s_eval = "../dataset/{}/result/Pic_Max_{}_D_{}_search_loss.pdf".format(D_tag, AF_tag, pic_name)
    draw_lines_loss(x, y, names, x_label="Iterations", y_label="Loss", P_title=pic_name, output_file=s_eval)


def compute_cumulative(input_data):
    """
    计算累计值
    :return:
    """
    output_data = []
    cum_data = 0
    for i in range(len(input_data)):
        if input_data[i]:
            cum_data += 1
            output_data.append(cum_data)
        else:
            output_data.append(cum_data)

    return output_data


def get_FCD_loss(model_file, test_file, dataset, protected, t1, t2, extent, N_cluster, N_seeds, check_tag):
    """
    检测进行 true bias， false bias， false fair 扰动时， 损失函数的变化情况
    :return:
    """
    model = keras.models.load_model(model_file)
    search_seeds = get_search_seeds(test_file, N_cluster, N_seeds)
    if check_tag in {"TB_G", "FB_G", "FF_G"}:
        item, loss = check_global_generation(model, search_seeds, dataset, protected, t1, extent, check_tag)
    else:
        item, loss = check_local_generation(model, search_seeds, dataset, protected, t2, extent, check_tag)

    return numpy.array(item), numpy.array(loss)


def get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, s_t, extent, N_cluster, N_seeds, C_tag):
    """
    AF 测试样本生成
    :return:
    """

    RF_R, RF_L = get_FCD_loss(M_file, T_file1, D_tag, P_attr, s_t, s_t, extent, N_cluster, N_seeds, C_tag)
    workbook_name = xlsxwriter.Workbook("../dataset/{}/result/Search_{}_Loss.xlsx".format(D_tag, C_tag))
    worksheet = workbook_name.add_worksheet("loss change")
    write_worksheet_2d_data(RF_L[:, :, 0], worksheet)
    worksheet = workbook_name.add_worksheet("loss change similar")
    write_worksheet_2d_data(RF_L[:, :, 1], worksheet)
    workbook_name.close()
