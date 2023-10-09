from utils.utils_evaluate import check_classification2

retrain_size = 5
if __name__ == "__main__":
    # "重训练模型评估"
    file1 = "../dataset/compas/data/test.npz.npy"
    file2 = "../dataset/compas/data/multiple_test.npz.npy"

    for cos in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for m in ["RobustFair"]:
            model_file = "../dataset/compas/retrain/{}_{}_{}.h5"
            model_files = [model_file.format(cos, m, n) for n in range(retrain_size)]
            eval_file = "../dataset/compas/result/Retrain_{}_{}_evaluation.xlsx".format(cos, m)
            check_classification2(model_files, file1, file2, eval_file)
