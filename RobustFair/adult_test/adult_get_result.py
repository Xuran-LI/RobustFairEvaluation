import xlsxwriter
from utils.utils_input_output import write_worksheet_2d_data
from utils.utils_result import combine_base_model_attack_evaluations, combine_retrain_model_evaluations, \
    combine_multi_seeds_attack_evaluations, combine_multi_iteration_local_evaluations, \
    combine_multi_iteration_global_evaluations

if __name__ == "__main__":
    # BL model attack generation
    D_tag = "adult"
    BL_attack_data = combine_base_model_attack_evaluations(D_tag)
    BL_attack_file = "../dataset/adult/result/Result_BL_attack.xlsx"
    workbook_name = xlsxwriter.Workbook(BL_attack_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(BL_attack_data, worksheet)
    workbook_name.close()

    # retrain model evaluation
    D_tag = "adult"
    retrain_eval = combine_retrain_model_evaluations(D_tag)
    retrain_eval_file = "../dataset/adult/result/Result_retrain_eval.xlsx"
    workbook_name = xlsxwriter.Workbook(retrain_eval_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(retrain_eval, worksheet)
    workbook_name.close()

    # BL model attack generation
    D_tag = "adult"
    seeds_attack_data = combine_multi_seeds_attack_evaluations(D_tag)
    seeds_attack_file = "../dataset/adult/result/Result_seeds_attack.xlsx"
    workbook_name = xlsxwriter.Workbook(seeds_attack_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(seeds_attack_data, worksheet)
    workbook_name.close()

    # BL model attack generation
    D_tag = "adult"
    iter_attack_data = combine_multi_iteration_local_evaluations(D_tag)
    iter_attack_file = "../dataset/adult/result/Result_iteration_Local.xlsx"
    workbook_name = xlsxwriter.Workbook(iter_attack_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(iter_attack_data, worksheet)
    workbook_name.close()

    # BL model attack generation
    D_tag = "adult"
    iter_attack_data = combine_multi_iteration_global_evaluations(D_tag)
    iter_attack_file = "../dataset/adult/result/Result_iteration_Global.xlsx"
    workbook_name = xlsxwriter.Workbook(iter_attack_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(iter_attack_data, worksheet)
    workbook_name.close()
