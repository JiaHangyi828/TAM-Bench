import pandas as pd

def evaluate_map_at_25(submission_path, label_path):
    # 加载文件
    submission = pd.read_csv(submission_path)
    labels = pd.read_csv(label_path)

    # 格式校验
    assert 'QuestionId_Answer' in submission.columns and 'MisconceptionId' in submission.columns, "submission.csv 格式错误"
    assert 'QuestionId_Answer' in labels.columns and 'MisconceptionId' in labels.columns, "test_labels.csv 格式错误"

    # 构建映射：ground truth
    label_dict = dict(zip(labels['QuestionId_Answer'], labels['MisconceptionId']))

    average_precisions = []

    for idx, row in submission.iterrows():
        qid_ans = row['QuestionId_Answer']
        preds = str(row['MisconceptionId']).strip().split()

        if qid_ans not in label_dict:
            continue  # submission中可能包含测试集没有的条目

        true_label = str(label_dict[qid_ans])

        # 找出正确标签在预测列表中的位置（从0开始）
        try:
            rank = preds.index(true_label)
            precision = 1.0 / (rank + 1)
        except ValueError:
            precision = 0.0

        average_precisions.append(precision)

    # 计算最终的 MAP@25
    if not average_precisions:
        return 0.0

    map25 = sum(average_precisions) / len(average_precisions)
    return map25


# 示例用法
if __name__ == "__main__":
    submission_file = "aideds/submission.csv"
    label_file = "test_labels.csv"
    map25 = evaluate_map_at_25(submission_file, label_file)
    print(f"MAP@25 Score: {map25:.6f}")
