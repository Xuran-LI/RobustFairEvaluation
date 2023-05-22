from utils.utils_draw import draw_lines

if __name__ == "__main__":
    dataset_name = "Bank Marketing"
    x = [100, 200, 300, 400, 500]
    false_n = [162.6, 357.7, 541.4, 689.6, 785.2]
    bias_n = [1209, 2378.1, 3518.6, 4588.5, 5859.2]
    false_or_bias_n = [1338.2, 2634.4, 3948.8, 5105.4, 6434.7]
    SUM = [1449.8, 2543.7, 4188.8, 5346.6, 6468.9]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/bank/result/Bank_seeds_evaluation.pdf"
    draw_lines(x, y, names, x_label="Seeds", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [950.5, 2692.6, 5074, 4069.6, 4390.8, 4187.6]
    bias_n = [1742.9, 3845.6, 7959.4, 6435.6, 6877.2, 6582.6]
    false_or_bias_n = [2346.1, 5410.6, 11237, 8969.6, 9754.6, 9114]
    SUM = [2543.7, 6028.4, 12350.8, 9837, 10831.4, 10125.2]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/bank/result/Bank_Local_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Local Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [950.5, 2233, 2482.4, 4557.2, 5002.4, 5843.8]
    bias_n = [1742.9, 3023.2, 4324, 6797.8, 6282.2, 8748]
    false_or_bias_n = [2346.1, 4528.6, 6161.8, 9729.6, 9960.6, 12582.4]
    SUM = [2543.7, 4926.4, 6658, 10553.8, 10784.8, 13474]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/bank/result/Bank_Global_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Global Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)
