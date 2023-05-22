from utils.utils_RobustFair import RobustFair_test

T_file = "../dataset/adult/data/test.npz.npy"
C_num = 4
D_tag = "adult"
P_attr = [10, 11, 12]
S_times = 5
E = 0.01

seeds_num = [100, 300, 400, 500]
if __name__ == "__main__":
    for S_num in seeds_num:
        print(S_num)
        M_names = ["BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL"]
        M_file = "../dataset/adult/model/{}.h5"
        M_files = [M_file.format(n) for n in M_names]

        D_file1 = "../dataset/adult/test/Seeds_" + str(S_num) + "_BL_{}_RF_generation.npz.npy"
        files1 = [D_file1.format(n) for n in range(len(M_names))]
        D_file2 = "../dataset/adult/test/Seeds_" + str(S_num) + "_BL_{}_RF_generation_similar.npz.npy"
        files2 = [D_file2.format(n) for n in range(len(M_names))]
        R_file = "../dataset/adult/result/Seeds_" + str(S_num) + "_BL_RF_generation_result.xlsx"

        RobustFair_test(M_files, T_file, C_num, S_num, D_tag, P_attr, S_times, S_times, E, files1, files2, R_file)
