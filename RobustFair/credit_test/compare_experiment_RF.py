from utils.utils_RobustFair import RobustFair_test

T_file = "../dataset/credit/data/test.npz.npy"
V_files = ["../dataset/credit/model/V_0.h5", "../dataset/credit/model/V_1.h5", "../dataset/credit/model/V_2.h5"]
C_num = 4
S_num = 200
D_tag = "credit"
P_attr = [19, 20]
S_times = 5
E = 0.01
if __name__ == "__main__":
    M_names = ["BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL"]
    M_file = "../dataset/credit/model/{}.h5"
    M_files = [M_file.format(n) for n in M_names]

    D_file1 = "../dataset/credit/test/BL_{}_RF_generation.npz.npy"
    files1 = [D_file1.format(n) for n in range(len(M_names))]
    D_file2 = "../dataset/credit/test/BL_{}_RF_generation_similar.npz.npy"
    files2 = [D_file2.format(n) for n in range(len(M_names))]
    R_file = "../dataset/credit/result/BL_RF_generation_result.xlsx"

    RobustFair_test(M_files, T_file, C_num, S_num, D_tag, P_attr, S_times, S_times, E, files1, files2, R_file)
