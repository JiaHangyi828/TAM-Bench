import pandas as pd

def load_and_check_format(test_labels_path, submission_path):
    test_df = pd.read_csv(test_labels_path)
    sub_df = pd.read_csv(submission_path)
    
    # 检查列名是否一致且顺序一致
    if list(test_df.columns) != list(sub_df.columns):
        print(f"格式错误：列名不一致\nTest labels列名: {list(test_df.columns)}\nSubmission列名: {list(sub_df.columns)}")
        return None, None, False
    
    # 检查row_id是否完全一致且顺序一致
    if not test_df['row_id'].equals(sub_df['row_id']):
        print("格式错误：row_id 不完全对应或顺序不同")
        # 可以具体列出不同的地方，方便调试
        diff_test = set(test_df['row_id']) - set(sub_df['row_id'])
        diff_sub = set(sub_df['row_id']) - set(test_df['row_id'])
        print(f"test_labels多出的row_id（不在submission中）: {diff_test}")
        print(f"submission多出的row_id（不在test_labels中）: {diff_sub}")
        return None, None, False
    
    return test_df, sub_df, True

def calc_micro_f1(test_df, sub_df):
    f1_scores = []
    
    for idx, (true_birds, pred_birds) in enumerate(zip(test_df['birds'], sub_df['birds'])):
        # 分割成集合
        true_set = set(true_birds.split()) if true_birds != "nocall" else set()
        pred_set = set(pred_birds.split()) if pred_birds != "nocall" else set()
        
        # 如果都为空集合，视为完全匹配，f1=1
        if not true_set and not pred_set:
            f1_scores.append(1.0)
            continue
        
        # 计算查准率、召回率
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    micro_f1 = sum(f1_scores) / len(f1_scores)
    return micro_f1

def evaluate(test_labels_path, submission_path):
    test_df, sub_df, valid = load_and_check_format(test_labels_path, submission_path)
    if not valid:
        print("提交文件格式校验未通过，无法计算分数。")
        return
    
    score = calc_micro_f1(test_df, sub_df)
    print(f"提交文件格式正确，micro F1 score: {score:.6f}")

if __name__ == "__main__":
    # 修改成你本地路径
    test_labels_path = "private/test_labels.csv"
    submission_path = "public/submission.csv"
    evaluate(test_labels_path, submission_path)
