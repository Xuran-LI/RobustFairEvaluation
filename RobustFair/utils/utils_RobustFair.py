import keras
import numpy
import xlsxwriter as xlsxwriter
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_evaluate import classification_evaluation1, check_item_AF
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data
from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad_MP, approximate_by_calculus_Max, \
    compute_loss_grad, sort_perturbation_direction, data_augmentation_compas_item, data_augmentation_credit_item, \
    data_augmentation_bank_item, data_augmentation_adult, data_augmentation_compas, data_augmentation_credit, \
    data_augmentation_bank


def false_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
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
            # max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
            # max_similar = similar_x_i[max_id].reshape(1, -1)
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
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            generated_y, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent,
                                                           direction)

            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def true_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
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
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            generated_y, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent,
                                                           direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def false_fair_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
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
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            generated_y, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent,
                                                           direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def global_generation(model, global_seeds, dataset, protected, search_times, extent):
    """
    全局搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    fb, fb_loss = false_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    tb, tb_loss = true_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    ff, ff_loss = false_fair_global(model, global_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((fb, tb, ff), axis=0)
    search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)

    unique_result, unique_index = numpy.unique(search_result, return_index=True, axis=0)
    unique_result_search_loss = search_loss[unique_index]

    return unique_result, unique_result_search_loss


def false_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)
        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
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
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def true_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
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
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def false_fair_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
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
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Max(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1), \
           numpy.squeeze(numpy.array(loss_perturbation))


def local_generation(model, local_seeds, dataset, protected, search_times, extent):
    """
    局部搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    fb, fb_loss = false_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    tb, tb_loss = true_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    ff, ff_loss = false_fair_local(model, local_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((fb, tb, ff), axis=0)
    search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)

    unique_result, unique_index = numpy.unique(search_result, return_index=True, axis=0)
    unique_result_search_loss = search_loss[unique_index]

    return unique_result, unique_result_search_loss


def retrain_RobustFair(model_file, seeds_file, dataset, protected, t1, t2, extent):
    """
    进行准确公平性测试
    :return:
    """
    model = keras.models.load_model(model_file)
    search_seeds = numpy.load(seeds_file)
    if t1 == 0:
        L_item, L_loss = local_generation(model, search_seeds, dataset, protected, t2, extent)
        RobustFair_result, unique_index = numpy.unique(L_item, return_index=True, axis=0)
    elif t2 == 0:
        G_item, G_loss = global_generation(model, search_seeds, dataset, protected, t1, extent)
        RobustFair_result, unique_index = numpy.unique(G_item, return_index=True, axis=0)
    else:
        G_item, G_loss = global_generation(model, search_seeds, dataset, protected, t1, extent)
        L_item, L_loss = local_generation(model, G_item, dataset, protected, t2, extent)
        search_item = numpy.concatenate((G_item, L_item), axis=0)
        RobustFair_result, unique_index = numpy.unique(search_item, return_index=True, axis=0)
    return RobustFair_result


def get_retrain_data_RobustFair(M_files, S_file, E_file, D_tag, P_attr, s_t1, s_t2, extent, f1, f2, R_file):
    """
    AF 测试样本生成
    :return:
    """
    evaulations = []
    for i in range(len(M_files)):
        RF_R = retrain_RobustFair(M_files[i], S_file, D_tag, P_attr, s_t1, s_t2, extent)
        numpy.save(f1[i], RF_R)
        if D_tag == "adult":
            numpy.save(f2[i], data_augmentation_adult(RF_R, P_attr))
        elif D_tag == "compas":
            numpy.save(f2[i], data_augmentation_compas(RF_R, P_attr))
        elif D_tag == "credit":
            numpy.save(f2[i], data_augmentation_credit(RF_R, P_attr))
        elif D_tag == "bank":
            numpy.save(f2[i], data_augmentation_bank(RF_R, P_attr))
        result_evaluation = classification_evaluation1(M_files[i], f1[i], f2[i], E_file)
        evaulations.append(result_evaluation)
        print(result_evaluation)

    header_name = ["Acc R", "acc N", "False R", "False N",
                   "IF R", "IF N", "IB R", "IB N",
                   "A&F R", "A&F", "F|B R", "F|B",
                   "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM",
                   "cos"]
    workbook_name = xlsxwriter.Workbook(R_file)
    worksheet = workbook_name.add_worksheet("Generation Details")
    write_worksheet_header(header_name, worksheet)
    write_worksheet_2d_data(evaulations, worksheet)
    workbook_name.close()
