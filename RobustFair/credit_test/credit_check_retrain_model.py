from utils.utils_evaluate import check_classification

retrain_size = 10
if __name__ == "__main__":
    # "重训练模型评估"
    file1 = "../dataset/credit/data/test.npz.npy"
    file2 = "../dataset/credit/data/multiple_test.npz.npy"

    # accurate fairness
    for n in ["RF"]:
        model_file = "../dataset/credit/retrain/" + n + "_{}.h5"
        eval_file = "../dataset/credit/result/Re_{}_evaluation.xlsx".format(n)
        model_files = [model_file.format(n) for n in range(retrain_size)]
        check_classification(model_files, file1, file2, eval_file)
