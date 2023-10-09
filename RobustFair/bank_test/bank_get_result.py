import xlsxwriter
from utils.utils_input_output import write_worksheet_2d_data
from utils.utils_result import combine_model_attack_evaluations10, combine_retrain_model_evaluations, \
    combine_model_attack_evaluations100

if __name__ == "__main__":
    D_tag = "bank"
    attack_data = combine_model_attack_evaluations10(D_tag)
    result_file = "../dataset/bank/result/Result_10_attack.xlsx"
    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(attack_data, worksheet)
    workbook_name.close()

    D_tag = "bank"
    attack_data = combine_model_attack_evaluations100(D_tag)
    result_file = "../dataset/bank/result/Result_100_attack.xlsx"
    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(attack_data, worksheet)
    workbook_name.close()

    D_tag = "bank"
    retrain_data = combine_retrain_model_evaluations(D_tag)
    result_file = "../dataset/bank/result/Result_retrain_eval.xlsx"
    workbook_name = xlsxwriter.Workbook(result_file)
    worksheet = workbook_name.add_worksheet("result Details")
    write_worksheet_2d_data(retrain_data, worksheet)
    workbook_name.close()

    for cos in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        D_tag = "bank"
        retrain_data = combine_retrain_model_evaluations(D_tag, cos)
        result_file = "../dataset/bank/result/Result_retrain_eval_{}.xlsx".format(cos)
        workbook_name = xlsxwriter.Workbook(result_file)
        worksheet = workbook_name.add_worksheet("result Details")
        write_worksheet_2d_data(retrain_data, worksheet)
        workbook_name.close()
