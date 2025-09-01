from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error

class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass

def grade(submission: DataFrame, answers: DataFrame) -> float:
    assert "id" in answers.columns, "Answers must have an 'id' column."
    assert "value" in answers.columns, "Answers must have a 'value' column."

    try:
        answers["value"].astype(float)
    except ValueError as e:
        assert False, "Answers must have float values in the 'value' column."

    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'id' column.")

    if "value" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'value' column.")

    try:
        submission["value"].astype(float)
    except ValueError as e:
        raise InvalidSubmissionError(f"Submission must have float values in the 'value' column.")

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected the submission to have {len(answers)} rows, but got {len(submission)}."
        )

    submission_sorted = submission.sort_values(by="id").sort_index(axis=1)
    answers_sorted = answers.sort_values(by="id").sort_index(axis=1)

    if (submission_sorted["id"].values != answers_sorted["id"].values).any():
        raise InvalidSubmissionError(
            "Expected the submission to have the same 'id' values as the answers, but they differ."
        )

    y_true = submission_sorted["value"]
    y_pred = answers_sorted["value"]
    score = root_mean_squared_error(y_true=y_true, y_pred=y_pred)

    return score

import pandas as pd
answers = pd.read_csv("answers.csv")

# 2. 读取用户提交的文件（预测值）
submission = pd.read_csv("openhandsds/submission.csv")

# 3. 调用评分函数
try:
    score = grade(submission, answers)
    print(f"✅ 评分成功！RMSE 得分: {score:.6f}")
except InvalidSubmissionError as e:
    print(f"❌ 提交无效：{e}")
except FileNotFoundError as e:
    print(f"📁 文件未找到：{e}")
except Exception as e:
    print(f"⚠️ 其他错误：{e}")
