from utils.utils_check_RobustFair import get_RobustFair_loss, check_RobustFair_loss

D_tag = "credit"
P_attr = [19, 20]
S_t = 50
E = 0.01

if __name__ == "__main__":
    M_file = "../dataset/credit/model/BL.h5"
    T_file1 = "../dataset/credit/data/test.npz.npy"
    T_file2 = "../dataset/credit/data/multiple_test.npz.npy"

    # 获取搜索过程中loss的折线图
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "TB_G", "TB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "FB_G", "FB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "FF_G", "FF Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "TB_L", "TB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "FB_L", "FB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TF", "FF_L", "FF Local Search")

    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "TB_G", "TB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "FB_G", "FB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "FF_G", "FF Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "TB_L", "TB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "FB_L", "FB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "TB", "FF_L", "FF Local Search")

    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "TB_G", "TB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "FB_G", "FB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "FF_G", "FF Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "TB_L", "TB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "FB_L", "FB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FB", "FF_L", "FF Local Search")

    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "TB_G", "TB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "FB_G", "FB Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "FF_G", "FF Global Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "TB_L", "TB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "FB_L", "FB Local Search")
    check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, S_t, E, "FF", "FF_L", "FF Local Search")

    # 获取搜索过程中loss的值
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "TB_G")
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "FB_G")
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "FF_G")
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "TB_L")
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "FB_L")
    get_RobustFair_loss(M_file, T_file1, D_tag, P_attr, S_t, E, 4, 100, "FF_L")
