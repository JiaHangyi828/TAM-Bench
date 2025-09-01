import argparse
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error
from pathlib import Path


def prepare_for_metric(submission: DataFrame, answers: DataFrame):
    required_columns = {"key", "fare_amount"}

    # 空文件或空数据
    if submission.empty:
        print("❌ 格式错误：提交文件为空")
        return None
    if answers.empty:
        print("❌ 格式错误：标签文件为空")
        return None

    # 缺少字段
    if not required_columns.issubset(submission.columns):
        print(f"❌ 格式错误：提交文件缺少字段，需包含 {required_columns}")
        return None
    if not required_columns.issubset(answers.columns):
        print(f"❌ 格式错误：标签文件缺少字段，需包含 {required_columns}")
        return None

    # key 是否有重复
    if submission["key"].duplicated().any():
        print("❌ 格式错误：提交文件中 key 存在重复项")
        return None
    if answers["key"].duplicated().any():
        print("❌ 格式错误：标签文件中 key 存在重复项")
        return None

    # key 是否匹配
    sub_keys = set(submission["key"])
    ans_keys = set(answers["key"])
    if sub_keys != ans_keys:
        print("❌ 格式错误：提交和标签的 key 不一致")
        return None

    # 排序对齐
    submission = submission.sort_values("key").reset_index(drop=True)
    answers = answers.sort_values("key").reset_index(drop=True)

    return answers["fare_amount"], submission["fare_amount"]

def grade(submission: DataFrame, answers: DataFrame) -> float:
    result = prepare_for_metric(submission, answers)
    if result is None:
        return None
    y_true, y_pred = result
    return root_mean_squared_error(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser(description="Evaluate submission against ground truth labels.")
    parser.add_argument('--submission', type=str, required=True, help="Path to submission CSV file")
    parser.add_argument('--labels', type=str, required=True, help="Path to ground truth labels CSV file")
    args = parser.parse_args()

    submission_path = Path(args.submission)
    labels_path = Path(args.labels)

    try:
        submission = pd.read_csv(submission_path)
        answers = pd.read_csv(labels_path)

        score = grade(submission, answers)
        if score is not None:
            print(f"✅ RMSE score: {score:.6f}")

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e.filename}")
    except pd.errors.EmptyDataError:
        print(f"❌ 格式错误：CSV 文件为空")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    main()
