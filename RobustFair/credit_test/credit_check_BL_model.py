from utils.utils_evaluate import check_classification

if __name__ == "__main__":
    # "检查模型的准确公平性"
    model_names = ["BL"]
    model_file = "../dataset/credit/model/{}.h5"
    model_files = [model_file.format(n) for n in model_names]
    file1 = "../dataset/credit/data/test.npz.npy"
    file2 = "../dataset/credit/data/multiple_test.npz.npy"
    eval_file = "../dataset/credit/result/BL_model_evaluation.xlsx"
    check_classification(model_files, file1, file2, eval_file)
