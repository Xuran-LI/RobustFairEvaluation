import pandas
import numpy


def adjust_ctrip_data(test_file):
    """
    获取ctrip的numerical数据
    :return:
    """
    data = pandas.read_csv(test_file, header=None).values
    hotel, user_service = numpy.split(data, [6, ], axis=1)
    return numpy.concatenate((user_service, hotel), axis=1)


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

