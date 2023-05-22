from utils.utils_RobustFair import RobustFair_test

T_file = "../dataset/compas/data/test.npz.npy"
C_num = 4
D_tag = "compas"
S_num = 200
P_attr = [0, 1, 2]
S_t1 = 5
E = 0.01

local_iterations = [10, 15, 20, 25, 30]
if __name__ == "__main__":
    for S_t2 in local_iterations:
        print(S_t2)
        M_names = ["BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL", "BL"]
        M_file = "../dataset/compas/model/{}.h5"
        M_files = [M_file.format(n) for n in M_names]

        D_file1 = "../dataset/compas/test/Iterations_" + str(S_t2) + "_BL_{}_RF_generation.npz.npy"
        files1 = [D_file1.format(n) for n in range(len(M_names))]
        D_file2 = "../dataset/compas/test/Iterations_" + str(S_t2) + "_BL_{}_RF_generation_similar.npz.npy"
        files2 = [D_file2.format(n) for n in range(len(M_names))]
        R_file = "../dataset/compas/result/Iterations_" + str(S_t2) + "_BL_RF_generation_result.xlsx"

        RobustFair_test(M_files, T_file, C_num, S_num, D_tag, P_attr, S_t1, S_t2, E, files1, files2, R_file)
