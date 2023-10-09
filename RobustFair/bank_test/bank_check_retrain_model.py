from utils.utils_evaluate import check_classification2

retrain_size = 1
if __name__ == "__main__":
    file1 = "../dataset/bank/data/test.npz.npy"
    file2 = "../dataset/bank/data/multiple_test.npz.npy"

    for cos in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for m in ["RobustFair"]:
            model_file = "../dataset/bank/retrain/{}_{}_{}.h5"
            model_files = [model_file.format(cos, m, n) for n in range(retrain_size)]
            eval_file = "../dataset/bank/result/Retrain_{}_{}_evaluation.xlsx".format(cos, m)
            check_classification2(model_files, file1, file2, eval_file)
