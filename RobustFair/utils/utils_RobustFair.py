import time
import keras
import numpy
import xlsxwriter as xlsxwriter
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_evaluate import classification_evaluation, check_item_AF
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data
from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad_MP, compute_label_Taylor_C, \
    compute_loss_grad, sort_perturbation_direction, get_global_seeds, data_augmentation_adult, far_similar_C, \
    data_augmentation_compas_item, data_augmentation_compas, data_augmentation_credit_item, data_augmentation_credit, \
    data_augmentation_bank_item, data_augmentation_bank


def false_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        for _ in range(search_times):
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)

            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)

            max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
            grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数 确定非保护属性扰动方向
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] > 0:
                        direction[0, n] = grad1[n]
                    elif sign1[n] < 0:
                        direction[0, n] = -1 * grad1[n]
                    elif sign1[n] == 0:
                        if sign2[n] > 0:
                            direction[0, n] = -1 * grad2[n]
                        elif sign2[n] < 0:
                            direction[0, n] = grad2[n]
            #  根据梯度方向进行扰动，计算生成的item
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据泰勒公式估计生成真实标记
            generated_y = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, generated_x, model)
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def true_bias_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        for _ in range(search_times):
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)

            y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)

            max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
            grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
            sign2 = numpy.sign(grad2)
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] > 0:
                        direction[0, n] = -1 * grad1[n]
                    elif sign1[n] < 0:
                        direction[0, n] = grad1[n]
                    elif sign1[n] == 0:
                        if sign2[n] > 0:
                            direction[0, n] = grad2[n]
                        elif sign2[n] < 0:
                            direction[0, n] = -1 * grad2[n]
            # 根据梯度方向进行扰动，计算扰动后的item，label
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据泰勒公式估计生成真实标记
            generated_y = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, generated_x, model)
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def false_fair_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        for _ in range(search_times):
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)

            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)

            max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
            grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
            sign2 = numpy.sign(grad2)
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] == sign2[n]:
                    if sign1[n] > 0:
                        direction[0, n] = grad1[n]
                    elif sign1[n] < 0:
                        direction[0, n] = -1 * grad1[n]
                    elif sign1[n] == 0:
                        direction[0, n] = grad2[n]
            #  根据梯度方向进行扰动，计算生成item，label
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据泰勒公式估计生成真实标记
            generated_y = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, generated_x, model)
            x_i = generated_x
            y_i = generated_y
            generate_x.append(x_i)
            generate_y.append(y_i)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def global_generation(model, global_seeds, dataset, protected, search_times, extent):
    """
    全局搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    Data_FB = false_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    Data_TB = true_bias_global(model, global_seeds, dataset, protected, search_times, extent)
    Data_FF = false_fair_global(model, global_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((Data_FB, Data_TB, Data_FF), axis=0)
    unique_search = numpy.unique(search_result, axis=0)
    return unique_search


def false_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

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

        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)

        max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
        grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
        sign2 = numpy.sign(grad2)
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] > 0:
                    direction[0, n] = grad1[n]
                elif sign1[n] < 0:
                    direction[0, n] = -1 * grad1[n]
                elif sign1[n] == 0:
                    if sign2[n] > 0:
                        direction[0, n] = -1 * grad2[n]
                    elif sign2[n] < 0:
                        direction[0, n] = grad2[n]
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
            pert_l = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, pert_x, model)
            generate_x.append(pert_x)
            generate_y.append(pert_l)

    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def true_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

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

        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        # 计算样本损失梯度, 计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)

        max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
        grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
        sign2 = numpy.sign(grad2)
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] > 0:
                    direction[0, n] = -1 * grad1[n]
                elif sign1[n] < 0:
                    direction[0, n] = grad1[n]
                elif sign1[n] == 0:
                    if sign2[n] > 0:
                        direction[0, n] = grad2[n]
                    elif sign2[n] < 0:
                        direction[0, n] = -1 * grad2[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            pert_l = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, pert_x, model)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def false_fair_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()

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

        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)

        max_id = far_similar_C(model.predict(similar_x_i), model.predict(x_i))
        grad2 = compute_loss_grad_MP(similar_x_i, y_cate, model, max_id)
        sign2 = numpy.sign(grad2)
        direction = numpy.zeros_like(x_i)

        for n in range(x.shape[1]):
            if n not in protected and sign1[n] == sign2[n]:
                if sign1[n] > 0:
                    direction[0, n] = grad1[n]
                else:
                    direction[0, n] = -1 * grad1[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            pert_l = compute_label_Taylor_C(x_i, y_cate, grad1, perturbation, pert_x, model)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
    return numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)


def local_generation(model, local_seeds, dataset, protected, search_times, extent):
    """
    局部搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    Data_FB = false_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    Data_TB = true_bias_local(model, local_seeds, dataset, protected, search_times, extent)
    Data_FF = false_fair_local(model, local_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((Data_FB, Data_TB, Data_FF), axis=0)
    unique_search = numpy.unique(search_result, axis=0)
    return unique_search


def RobustFair_classifier(model_file, test_file, cluster_num, sample_num, dataset, protected, s_t1, s_t2, extent):
    """
    进行准确公平性测试
    :return:
    """
    t1 = time.time()
    model = keras.models.load_model(model_file)
    global_seeds = get_global_seeds(test_file, cluster_num, sample_num)
    local_seeds = global_generation(model, global_seeds, dataset, protected, s_t1, extent)
    RF_results = local_generation(model, local_seeds, dataset, protected, s_t2, extent)
    t2 = time.time()
    return RF_results, t2 - t1


def RobustFair_test(M_files, T_file, C_num, S_num, D_tag, P_attr, s_t1, s_t2, extent, files1, files2, R_file):
    """
    AF 测试样本生成
    :return:
    """
    Taylor_eval = []
    for i in range(len(M_files)):
        RF_result, RF_time = RobustFair_classifier(M_files[i], T_file, C_num, S_num, D_tag, P_attr, s_t1, s_t2, extent)
        numpy.save(files1[i], RF_result)
        if D_tag == "adult":
            numpy.save(files2[i], data_augmentation_adult(RF_result, P_attr))
        elif D_tag == "compas":
            numpy.save(files2[i], data_augmentation_compas(RF_result, P_attr))
        elif D_tag == "credit":
            numpy.save(files2[i], data_augmentation_credit(RF_result, P_attr))
        elif D_tag == "bank":
            numpy.save(files2[i], data_augmentation_bank(RF_result, P_attr))
        result_evaluation = classification_evaluation(M_files[i], files1[i], files2[i])

        Taylor_eval.append(result_evaluation)
        print(result_evaluation)

    header_name = ["avg", "std", "acc R", "false R", "acc N", "false N", "SUM",
                   "IFR", "IBR", "IFN", "IBN", "SUM",
                   "A&F R", "F|B R", "A&F", "F|B", "SUM",
                   "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM"]
    workbook_name = xlsxwriter.Workbook(R_file)
    worksheet = workbook_name.add_worksheet("Generation Details")
    write_worksheet_header(header_name, worksheet)
    write_worksheet_2d_data(Taylor_eval, worksheet)
    workbook_name.close()
