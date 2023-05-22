from utils.utils_draw import draw_lines

if __name__ == "__main__":
    dataset_name = "German Credit"
    x = [100, 200, 300, 400, 500]
    false_n = [1003, 2481.9, 2770.2, 3414.3, 4239.5]
    bias_n = [2363.6, 4758.5, 7026, 8649.8, 11055.5]
    false_or_bias_n = [2575, 5222.7, 7588, 9414.8, 11849.4]
    SUM = [2607.2, 5303, 7728.2, 9607, 12105.8]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/credit/result/Credit_seeds_evaluation.pdf"
    draw_lines(x, y, names, x_label="Seeds", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [2481.9, 3392.8, 5554.8, 7076.2, 8031.4, 7827.2]
    bias_n = [4758.5, 8442.8, 13969.8, 17016, 17923.2, 16737.2]
    false_or_bias_n = [5222.7, 9494, 15548.2, 19492.6, 20071.6, 19758]
    SUM = [5303, 9750.8, 15870.8, 19824.6, 20493.2, 20137.2]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/credit/result/Credit_Local_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Local Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [2481.9, 3828.4, 5408.6, 7404, 7573.4, 10369]
    bias_n = [4758.5, 9242.2, 13168, 16265.6, 17548.4, 24531.2]
    false_or_bias_n = [5222.7, 10017.6, 14573.4, 18198.6, 20105.4, 26750.4]
    SUM = [5303, 10179.8, 14790.6, 18460.4, 20404, 27126]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/credit/result/Credit_Global_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Global Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)
