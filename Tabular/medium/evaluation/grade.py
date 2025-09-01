import pandas as pd
import sys
from sklearn.metrics import matthews_corrcoef

def evaluate_submission(test_label_path='test_label.csv', submission_path='submission.csv'):
    # 读取文件
    try:
        test_df = pd.read_csv(test_label_path)
    except Exception as e:
        print("Error reading test_label.csv:", e)
        return

    try:
        sub_df = pd.read_csv(submission_path)
    except Exception as e:
        print("Error reading submission.csv:", e)
        return

    # 检查列名是否一致
    if list(test_df.columns) != list(sub_df.columns):
        print("格式错误：列名与 test_label.csv 不一致")
        print(f"test_label.csv 列名: {list(test_df.columns)}")
        print(f"submission.csv 列名: {list(sub_df.columns)}")
        return

    # 假设第一列是 'id'，第二列是 'class'
    id_col, class_col = test_df.columns[0], test_df.columns[1]

    # 检查 id 是否一一对应（顺序可能不同，但必须完全匹配）
    test_ids = set(test_df[id_col])
    sub_ids = set(sub_df[id_col])

    if test_ids != sub_ids:
        print("格式错误：submission.csv 中的 id 与 test_label.csv 不匹配")
        missing_in_sub = test_ids - sub_ids
        extra_in_sub = sub_ids - test_ids
        if missing_in_sub:
            print(f"以下 id 缺失: {sorted(missing_in_sub)[:10]} {'...' if len(missing_in_sub) > 10 else ''}")
        if extra_in_sub:
            print(f"以下 id 多余: {sorted(extra_in_sub)[:10]} {'...' if len(extra_in_sub) > 10 else ''}")
        return

    # 确保 submission 中的顺序与 test_label 一致（按 id 排序）
    test_df = test_df.sort_values(by=id_col).reset_index(drop=True)
    sub_df = sub_df.sort_values(by=id_col).reset_index(drop=True)

    # 再次确认 id 完全对齐
    if not (test_df[id_col].equals(sub_df[id_col])):
        print("格式错误：id 无法一一对应（排序后仍不一致）")
        return

    # 获取真实标签和预测标签
    y_true = test_df[class_col]
    y_pred = sub_df[class_col]

    # 检查类别是否有效（假设类别是像 'e', 'p' 这样的字符串）
    # 可选：检查是否为字符串类型
    if not y_pred.apply(lambda x: isinstance(x, str)).all():
        print("格式错误：class 列包含非字符串值")
        return

    # 计算 MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"得分: {mcc:.6f}")
    except Exception as e:
        print("计算 MCC 时出错:", e)
        return

# 使用方式
if __name__ == "__main__":
    evaluate_submission(r'E:\Users\Administrator\Desktop\experiment\Tabular\medium\Binary Prediction of Poisonous Mushrooms\data\private\test_labels.csv', "openhandsds/submission.csv")