import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import pandas as pd
from numpy import ndarray
from scipy.special import softmax

class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass

def one_hot_dfs_to_log_loss_inputs(
    submission_one_hot: pd.DataFrame,
    answers_one_hot: pd.DataFrame,
    id_column: str = "id",
    apply_softmax: bool = True,
) -> dict:
    """
    Frequently used logic to prepare one-hotted dfs for log loss calculation.
    """
    required_cols = set(answers_one_hot.columns)
    submission_cols = set(submission_one_hot.columns)

    if not submission_cols.issuperset(required_cols):
        raise InvalidSubmissionError(
            f"The submission DataFrame is missing some columns required by the `answers` DataFrame. "
            f"Missing columns: {required_cols - submission_cols}."
        )
    if id_column not in submission_one_hot.columns:
        raise InvalidSubmissionError(f"Submission is missing id column '{id_column}'.")

    assert id_column in answers_one_hot.columns, f"Answers is missing id column '{id_column}'."

    # Filter submission to only include columns that are in the answers
    submission_filtered = submission_one_hot[
        [col for col in answers_one_hot.columns if col in submission_cols]
    ]

    # Sort submission and answers by id to align them
    submission_sorted = submission_filtered.sort_values(by=id_column).reset_index(drop=True)
    answers_sorted = answers_one_hot.sort_values(by=id_column).reset_index(drop=True)

    assert submission_sorted[id_column].tolist() == answers_sorted[id_column].tolist(), (
        f"Mismatch in {id_column.capitalize()}s between `submission` and `answers` after sorting. "
        f"Number of mismatched {id_column.capitalize()}s: {len(set(submission_sorted[id_column]) ^ set(answers_sorted[id_column]))}. "
        f"Ensure both DataFrames have the same {id_column.capitalize()}s."
    )

    assert list(submission_sorted.columns) == list(answers_sorted.columns), (
        "Column order mismatch after filtering and sorting. "
        "Ensure both DataFrames have columns in the same order."
    )

    y_true = answers_sorted.drop(columns=[id_column]).to_numpy()
    y_pred = submission_sorted.drop(columns=[id_column]).to_numpy()

    if apply_softmax and is_one_hot_encoded(y_pred):
        print(
            "The flag `apply_softmax` has been set to `True` but the submission is already "
            "one-hot encoded. Skipping softmax."
        )

    if apply_softmax and not is_one_hot_encoded(y_pred):
        y_pred = softmax(y_pred, axis=-1)

    log_loss_inputs = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return log_loss_inputs


def is_one_hot_encoded(xs: ndarray) -> bool:
    """Check if a 2D NumPy array is one-hot encoded."""

    assert isinstance(xs, ndarray), f"Expected a NumPy array, got {type(xs)}."
    assert xs.ndim == 2, f"Expected a 2D array, got {xs.ndim}D."

    is_binary_matrix = np.bitwise_or(xs == 0, xs == 1).all()
    is_normalized = np.allclose(xs.sum(axis=-1), 1)
    is_one_hot = bool(is_binary_matrix and is_normalized)

    assert isinstance(is_one_hot, bool), f"Expected a boolean, got {type(is_one_hot)}."

    return is_one_hot


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """
    The submission and answers are already one-hotted
    """
    classes = ["winner_model_a", "winner_model_b", "winner_tie"]
    required_columns = ["id"] + classes

    # Check if submission has the required columns
    missing_columns = [col for col in required_columns if col not in submission.columns]
    if missing_columns:
        raise InvalidSubmissionError(
            f"Submission DataFrame is missing required columns: {missing_columns}"
        )

    # Check if answers has the required columns
    assert set(required_columns).issubset(
        answers.columns
    ), f"Answers DataFrame is missing required columns: {set(required_columns) - set(answers.columns)}"

    # Check if submission has the correct number of rows
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission DataFrame must have {len(answers)} rows, but has {len(submission)} rows."
        )

    # Check if all values in submission are between 0 and 1
    if (
        not ((submission[classes] >= 0) & (submission[classes] <= 1)).all().all()
    ):  # first all() checks if all rows are valid, second all() checks if all columns are valid
        raise InvalidSubmissionError("All values in submission DataFrame must be between 0 and 1.")

    # Check if each row in submission sums to 1
    if not submission[classes].sum(axis=1).round(6).eq(1).all():
        raise InvalidSubmissionError("Each row in submission DataFrame must sum to 1.")

    # Use only the required columns for further processing
    submission = submission[required_columns]
    answers = answers[required_columns]

    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission and answer IDs do not match after sorting.")

    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column="id", apply_softmax=False
    )

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)



submission = pd.read_csv("openhandsds/submission.csv")
answers = pd.read_csv("answers.csv")

score = grade(submission, answers)
print(f"LogLoss: {score:.6f}")
