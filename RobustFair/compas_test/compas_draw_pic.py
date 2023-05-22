from utils.utils_draw import draw_lines

if __name__ == "__main__":
    dataset_name = "ProPublica Recidivism"
    x = [100, 200, 300, 400, 500]
    false_n = [593.2, 950.5, 1733.9, 2224.3, 2943.4]
    bias_n = [993.3, 1742.9, 2892.5, 3667, 4228]
    false_or_bias_n = [1358.8, 2346.1, 3904.7, 4970.4, 6024.5]
    SUM = [1667.6, 3221.1, 4769.5, 6268.3, 7797.5]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/compas/result/COMPAS_seeds_evaluation.pdf"
    draw_lines(x, y, names, x_label="Seeds", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [357.7, 733, 709.8, 824.6, 383.6, 704]
    bias_n = [2378.1, 4701.4, 4118.8, 4589.6, 4203.4, 3799.8]
    false_or_bias_n = [2634.4, 5165, 4596.8, 5187.8, 4474, 4320]
    SUM = [3221.1, 6303.8, 5532.2, 6247.6, 5426, 5317]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/compas/result/COMPAS_Local_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Local Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)

    x = [5, 10, 15, 20, 25, 30]
    false_n = [357.7, 462.8, 921.4, 1151.2, 1595, 2282.6]
    bias_n = [2378.1, 5141.6, 7456.4, 9187.6, 11492, 14702.4]
    false_or_bias_n = [2634.4, 5488.4, 8051.6, 10006, 12911.2, 16413.8]
    SUM = [3221.1, 6723.2, 9786.4, 12189, 15688.8, 20160.4]
    y = [false_n, bias_n, false_or_bias_n, SUM]
    names = ["False", "Biased", "False or Biased", "SUM"]
    s_eval = "../dataset/compas/result/COMPAS_Global_iter_evaluation.pdf"
    draw_lines(x, y, names, x_label="Global Iterations", y_label="Instances", P_title=dataset_name, output_file=s_eval)
