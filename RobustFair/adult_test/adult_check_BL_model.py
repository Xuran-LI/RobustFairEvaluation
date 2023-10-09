from utils.utils_evaluate import check_classification2

if __name__ == "__main__":
    model_names = ["BL"]
    model_file = "../dataset/adult/model/{}.h5"
    model_files = [model_file.format(n) for n in model_names]
    file1 = "../dataset/adult/data/test.npz.npy"
    file2 = "../dataset/adult/data/multiple_test.npz.npy"
    eval_file = "../dataset/adult/result/Train_BL_evaluation.xlsx"
    check_classification2(model_files, file1, file2, eval_file)
