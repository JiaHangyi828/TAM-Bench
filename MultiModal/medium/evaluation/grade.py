import pandas as pd
import ast

def evaluate_mrr_by_order(submission_path, groundtruth_path, k=100):
    # 加载数据
    submission_df = pd.read_csv(submission_path)
    groundtruth_df = pd.read_csv(groundtruth_path)

    # 基础校验
    assert len(submission_df) == len(groundtruth_df), "行数不一致：submission 和 groundtruth 数量不同"

    total_rr = 0.0
    num_samples = len(submission_df)

    for i in range(num_samples):
        preds = ast.literal_eval(submission_df.iloc[i]["predictions"])[:k]
        true_item = groundtruth_df.iloc[i]["next_item"]

        try:
            rank = preds.index(true_item) + 1  # 排名从1开始
            rr = 1.0 / rank
        except ValueError:
            rr = 0.0  # 没有命中

        total_rr += rr

    mrr = total_rr / num_samples
    print(f"MRR@{k}: {mrr:.6f}")
    return mrr

# 示例用法：
evaluate_mrr_by_order("openhandsgpt/submission.csv", "gt_task1.csv", k=100)
