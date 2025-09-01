from os.path import join
import json
from sklearn.metrics import roc_auc_score
import argparse

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

'''
系统评测脚本执行命令：
python grade.py -hp ind_valid_author_submit.json -rf ind_valid_author_ground_truth.json -l tmp_log.txt
python your_script.py -hp hypothesis.csv -rf reference.csv -l result.log
其中: your_script.py -hp [学生提交文件] -rf [答案文件] -l [结果文件]
结果文件内容格式：A）如果成功评测. 将分数与附加信息以 ### (三个井号)分隔写入结果文件. e.g. 0.938100###Submission success p=0.842729 r=0.911891 排行榜会根据分数进行排名，附加信息会显示在学生的“我的提交”页面. B) 如果评测失败. 直接将错误信息写入结果文件即可. 学生可以在‘我的提交’页面看到错误信息，并根据该信息重新尝试提交.
请注意：我们目前只接受python3脚本.
'''


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def format_check(submit_fname, gt_fname):
    data_dir = "./"
    flag = True
    info_json = {"err_code": 0, "err_msg": "[success] submission success"}

    try:
        data_dict = load_json(data_dir, submit_fname)
    except Exception as e:
        with open(args.l, "w", encoding="utf-8") as f:
            error_code = 1
            err_msg = "JSON load error"
            other_info = str(e)
            info_json = {"error_code": error_code, "err_msg": err_msg, "other_info": other_info}
            return False, info_json
        
    labels_dict = load_json(data_dir, gt_fname)

    for aid in labels_dict:
        cur_normal_data = labels_dict[aid]["normal_data"]
        cur_outliers = labels_dict[aid]["outliers"]
        for item in cur_normal_data + cur_outliers:
            if aid not in data_dict:
                error_code = 2
                err_msg = "Author ID not in submission file"
                other_info = "Author ID: " + aid
                info_json = {"error_code": error_code, "err_msg": err_msg, "other_info": other_info}
                return False, info_json
            elif item not in data_dict[aid]:
                error_code = 3
                err_msg = "Paper ID not in author profile in submission file"
                other_info = "Author ID: " + aid + " Paper ID: " + item
                info_json = {"error_code": error_code, "err_msg": err_msg, "other_info": other_info}
                return False, info_json
            else:
                try:
                    v = float(data_dict[aid][item])
                except Exception as e:
                    error_code = 4
                    err_msg = "Value error (Not a number)"
                    other_info = "Author ID: " + aid + " Paper ID: " + item
                    info_json = {"error_code": error_code, "err_msg": err_msg, "other_info": other_info}
                    return False, info_json
    
    return flag, info_json


def cal_overall_auc(submit_fname, gt_fname, log_fname):
    data_dir = "./"
    flag, info_json = format_check(submit_fname, gt_fname)
    if not flag:
        with open(log_fname, "w", encoding="utf-8") as f:
            f.writelines(str(info_json))
        return 0

    data_dict = load_json(data_dir, submit_fname)
    labels_dict = load_json(data_dir, gt_fname)

    total_w = 0
    total_auc = 0
    for aid in labels_dict:
        cur_normal_data = labels_dict[aid]["normal_data"]
        cur_outliers = labels_dict[aid]["outliers"]
        cur_labels = []
        cur_preds = []
        cur_w = len(cur_outliers)
        for item in cur_normal_data:
            cur_labels.append(1)
            cur_preds.append(data_dict[aid][item])
            # cur_preds.append(1)
        for item in cur_outliers:
            cur_labels.append(0)
            cur_preds.append(data_dict[aid][item])
            # cur_preds.append(0)
        cur_auc = roc_auc_score(cur_labels, cur_preds)
        total_auc += cur_w * cur_auc
        total_w += cur_w
    avg_auc = total_auc / total_w
    with open(log_fname, "w", encoding="utf-8") as f:
        f.writelines(str(avg_auc) + f"###submision success")
    return 0


parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('-hp', help='学生提交文件')
parser.add_argument('-rf',  help='答案文件')
parser.add_argument('-l',  help='结果文件')
args = parser.parse_args()


if __name__ == "__main__":
    try:
        auc = cal_overall_auc(args.hp, args.rf, args.l)
        print(auc)
    except Exception as e:
        with open(args.l, "w", encoding="utf-8") as f:
            f.write(str(e))
