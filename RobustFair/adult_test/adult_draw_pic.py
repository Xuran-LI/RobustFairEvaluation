from utils.utils_draw import draw_lines

if __name__ == "__main__":
    dataset_name = "Adult"
    x = [100, 200, 300, 400, 500]
    false_n = [1041.3, 2093.3, 3206.7, 4003.2, 5322.8]
    bias_n = [1701.5, 3623.5, 5579, 7128.9, 8914.8]
    false_or_bias_n = [2123.6, 4577.7, 7021.2, 9016.5, 11571.3]
    SUM = [2293.7, 4925.7, 7482.3, 9691.9, 12373.1]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/adult/result/Adult_seeds_evaluation.pdf"
    draw_lines(x, y, names, x_label="Seeds", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [2093.3, 5776.8, 5539.2, 5759.6, 5292.2, 5786.2]
    bias_n = [3623.5, 10918.6, 11050.2, 10549, 9844.4, 10481.4]
    false_or_bias_n = [4577.7, 13577.2, 13399.8, 13066.2, 12316, 12597.6]
    SUM = [4925.7, 14533.4, 14323.8, 13895, 13227.6, 13522.2]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/adult/result/Adult_Local_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Local Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [2093.3, 3764, 5224, 6920.8, 8936.4, 10391.2]
    bias_n = [3623.5, 6300.6, 9067.6, 12857.2, 15517.4, 19627.8]
    false_or_bias_n = [4577.7, 8007.2, 11884.8, 16047, 19839.4, 25003.6]
    SUM = [4925.7, 8570.8, 12733.2, 17135.4, 21158.2, 26682]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/adult/result/Adult_Global_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Global Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)
