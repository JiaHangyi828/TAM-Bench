from sklearn.metrics import average_precision_score
import json
import argparse
import warnings
from lxml import etree

warnings.filterwarnings('ignore')
'''
System profiling script execution command：
python grade.py -hp submission_example_valid.json -rf ground_truths_valid.json -l result.log
wherein: grade.py -hp [submit document] -rf [answer file] -l [resulting file]
If the evaluation is successful, it will be written in the file:f"user({your_user_name}):map###{your_map}###submit successful!"
If there is a problem, write the corresponding problem information.
'''

def get_real_and_result_dic(real_path, answer_path):
    with open(real_path, 'r') as read_file:
        real_dic = json.load(read_file)
    with open(answer_path, 'r') as read_file:
        result_dic = json.load(read_file)
    return real_dic, result_dic


def _error(error_info):
    write_file.write(error_info + '\n')
    write_file.flush()
    print("An error has occurred. Please check the error information in the log file!")


def calculate_map(result_dic, real_dic):
    map_list = []
    for item in result_dic.keys():
        result_list = result_dic[item]
        for i in range(len(result_list)):
            if result_list[i] > 1 or result_list[i] < 0:
                _error(f"err_code: 1, err_msg: The confidence score does not belong to [0,1]")
                return None
            # if result_list[i] >= 0.5:
            #     result_list[i] = 1
            # else:
            #     result_list[i] = 0
        try:
            real_list = real_dic[item]
        except KeyError:
            _error(f"err_code: 2, err_msg: paper ID {item} does not in evaluted paper set.")
            return None
        try:
            # print(item, len(real_list), len(result_list))
            map_list.append(average_precision_score(real_list, result_list))
        except ValueError:
            _error(f"err_code: 0, the number of references of paper ID {item} mismatches with ground truths.")
            return None
    map_ = sum(map_list) / len(map_list)
    write_file.write(str(map_) + f"###submision success map={map_}\n")
    return None


parser = argparse.ArgumentParser(description='input argparse')
parser.add_argument('-hp', help='submit file', default='test.json')
parser.add_argument('-rf', help='answer file', default='real_dic.json')
parser.add_argument('-l', help='result file', default='result.log')
args = parser.parse_args()

if __name__ == '__main__':
    """
    using: python3 eval_f1.py -hp result.txt -rf true_result.txt -l res.log
    """
    write_file = open(args.l, 'w')
    real_dic, result_dic = get_real_and_result_dic(args.rf, args.hp)
    calculate_map(result_dic, real_dic)
    write_file.close()
