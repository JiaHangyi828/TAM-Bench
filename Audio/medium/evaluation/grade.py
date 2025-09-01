import pandas as pd
from sklearn.metrics import accuracy_score

class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass

def prepare_for_accuracy_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict:

    # Answers checks
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # sort on id to ensure correct order
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    y_pred = submission[target_column].to_numpy()
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="label", id_column="fname"
    )
    return accuracy_score(**accuracy_inputs)

import pandas as pd
answers = pd.read_csv("test.csv")
submission = pd.read_csv("openhandsgpt/submission.csv")

# 3. 调用评分函数
try:
    score = grade(submission, answers)
    print(f"✅ 评分成功！accuracy_score: {score:.6f}")
except InvalidSubmissionError as e:
    print(f"❌ 提交无效：{e}")
except FileNotFoundError as e:
    print(f"📁 文件未找到：{e}")
except Exception as e:
    print(f"⚠️ 其他错误：{e}")
