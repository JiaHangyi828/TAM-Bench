import pandas as pd
import numpy as np
import sys
from pathlib import Path

def evaluate_submission(submission_path, test_labels_path):
    # 1. 读取文件
    submission = pd.read_csv(submission_path, dtype={"id": str})
    groundtruth = pd.read_csv(test_labels_path, dtype={"id": str})

    # 2. 检查列名是否一致
    required_columns = ["id", "X4", "X11", "X18", "X26", "X50", "X3112"]
    if list(submission.columns) != required_columns:
        raise ValueError(f"submission.csv 列名应为: {required_columns}，但检测到: {list(submission.columns)}")

    # 3. 检查 id 是否一一对应（顺序一致）
    if not submission["id"].equals(groundtruth["id"]):
        raise ValueError("submission.csv 的 id 列与 test_labels.csv 不一致或顺序错乱")

    # 4. 计算 R²
    r2_scores = {}
    for trait in required_columns[1:]:
        y_true = groundtruth[trait].values
        y_pred = submission[trait].values

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
        r2_scores[trait] = r2 if r2 > 0 else 0.0

    # 5. 平均 R²
    mean_r2 = np.mean(list(r2_scores.values()))

    print("各性状 R²（小于 0 的视为 0）:")
    for k, v in r2_scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n📊 平均 R² (mean R²): {mean_r2:.4f}")

    return mean_r2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python evaluate.py <submission.csv> <test_labels.csv>")
        sys.exit(1)

    submission_path = Path(sys.argv[1])
    test_labels_path = Path(sys.argv[2])
    evaluate_submission(submission_path, test_labels_path)

# python evaluate.py raw/public/submission.csv raw/private/test_labels.csv