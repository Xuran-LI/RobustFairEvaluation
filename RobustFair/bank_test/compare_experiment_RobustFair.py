from utils.utils_RobustFair import get_retrain_data_RobustFair

D_tag = "bank"
P_attr = [0]
E = 0.01
if __name__ == "__main__":
    M_names = ["BL"]
    S_file = "../dataset/bank/test/test_seeds.npz.npy"
    E_file = "../dataset/bank/test/test_avg_clusters.npz.npy"
    M_file = "../dataset/bank/model/{}.h5"
    M_files = [M_file.format(n) for n in M_names]
    D_file1 = "../dataset/bank/test/Test_RobustFair_{}_100_0.npz.npy"
    files1 = [D_file1.format(n) for n in range(len(M_names))]
    D_file2 = "../dataset/bank/test/Test_RobustFair_{}_100_0_similar.npz.npy"
    files2 = [D_file2.format(n) for n in range(len(M_names))]
    R_file = "../dataset/bank/result/Test_RobustFair_100_0.xlsx"
    get_retrain_data_RobustFair(M_files, S_file, E_file, D_tag, P_attr, 10, 0, E, files1, files2, R_file)

    M_names = ["BL"]
    S_file = "../dataset/bank/test/test_seeds.npz.npy"
    E_file = "../dataset/bank/test/test_avg_clusters.npz.npy"
    M_file = "../dataset/bank/model/{}.h5"
    M_files = [M_file.format(n) for n in M_names]
    D_file1 = "../dataset/bank/test/Test_RobustFair_{}_0_100.npz.npy"
    files1 = [D_file1.format(n) for n in range(len(M_names))]
    D_file2 = "../dataset/bank/test/Test_RobustFair_{}_0_100_similar.npz.npy"
    files2 = [D_file2.format(n) for n in range(len(M_names))]
    R_file = "../dataset/bank/result/Test_RobustFair_0_100.xlsx"
    get_retrain_data_RobustFair(M_files, S_file, E_file, D_tag, P_attr, 0, 10, E, files1, files2, R_file)

    M_names = ["BL"]
    S_file = "../dataset/bank/test/test_seeds.npz.npy"
    E_file = "../dataset/bank/test/test_avg_clusters.npz.npy"
    M_file = "../dataset/bank/model/{}.h5"
    M_files = [M_file.format(n) for n in M_names]
    D_file1 = "../dataset/bank/test/Test_RobustFair_{}_10_10.npz.npy"
    files1 = [D_file1.format(n) for n in range(len(M_names))]
    D_file2 = "../dataset/bank/test/Test_RobustFair_{}_10_10_similar.npz.npy"
    files2 = [D_file2.format(n) for n in range(len(M_names))]
    R_file = "../dataset/bank/result/Test_RobustFair_10_10.xlsx"
    get_retrain_data_RobustFair(M_files, S_file, E_file, D_tag, P_attr, 10, 10, E, files1, files2, R_file)

    M_names = ["BL"]
    S_file = "../dataset/bank/test/retrain_seeds.npz.npy"
    E_file = "../dataset/bank/test/retrain_avg_clusters.npz.npy"
    M_file = "../dataset/bank/model/{}.h5"
    M_files = [M_file.format(n) for n in M_names]
    D_file1 = "../dataset/bank/test/Train_RobustFair_{}_10_10.npz.npy"
    files1 = [D_file1.format(n) for n in range(len(M_names))]
    D_file2 = "../dataset/bank/test/Train_RobustFair_{}_10_10_similar.npz.npy"
    files2 = [D_file2.format(n) for n in range(len(M_names))]
    R_file = "../dataset/bank/result/Train_RobustFair_10_10.xlsx"
    get_retrain_data_RobustFair(M_files, S_file, E_file, D_tag, P_attr, 10, 10, E, files1, files2, R_file)
