def write_worksheet_2d_data(data, worksheet):
    """
    输出2维数据至worksheet
    :return:
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            worksheet.write(i + 1, j, data[i][j])


def write_worksheet_header(headers, worksheet):
    """
    输出header至worksheet
    :return:
    """
    for i in range(len(headers)):
        worksheet.write(0, i, headers[i])
