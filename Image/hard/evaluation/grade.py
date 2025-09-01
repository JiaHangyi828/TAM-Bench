import numpy as np
import pandas as pd
import itertools
import sys
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd

def load_submission(file_path):
    df = pd.read_csv(file_path)
    required_columns = ['image_id'] + [f'R{i}{j}' for i in range(3) for j in range(3)] + ['T0', 'T1', 'T2']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f'Submission missing required columns: {required_columns}')
    return df

def load_ground_truth(file_path):
    df = pd.read_csv(file_path)
    required_columns = ['image_id'] + [f'R{i}{j}' for i in range(3) for j in range(3)] + ['T0', 'T1', 'T2']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f'Ground truth missing required columns: {required_columns}')
    return df

def get_camera_centers(df):
    centers = {}
    for _, row in df.iterrows():
        R_matrix = np.array(row[[f'R{i}{j}' for i in range(3) for j in range(3)]]).reshape(3, 3)
        T_vec = np.array(row[['T0', 'T1', 'T2']]).reshape(3, 1)
        C = -R_matrix.T @ T_vec
        centers[row['image_id']] = C.flatten()
    return centers

def compute_similarity_transform(src, dst):
    # Based on Umeyama method
    src = np.array(src).T  # 3×N
    dst = np.array(dst).T
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    H = src_centered @ dst_centered.T
    U, _, Vt = svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[2, :] *= -1
        R_opt = Vt.T @ U.T
    scale = np.trace(R_opt @ H) / np.trace(src_centered @ src_centered.T)
    t_opt = mean_dst - scale * R_opt @ mean_src
    return scale, R_opt, t_opt

def apply_similarity_transform(C, scale, R_opt, t_opt):
    return scale * R_opt @ C + t_opt

def evaluate(pred_centers, gt_centers, threshold=0.5):
    common_ids = set(pred_centers.keys()) & set(gt_centers.keys())
    best_inliers = 0
    best_transform = None
    for triplet in itertools.combinations(common_ids, 3):
        src = [pred_centers[i] for i in triplet]
        dst = [gt_centers[i] for i in triplet]
        try:
            scale, R_opt, t_opt = compute_similarity_transform(src, dst)
        except:
            continue
        inliers = 0
        for img_id in common_ids:
            pred_C = pred_centers[img_id].reshape(3, 1)
            pred_transformed = apply_similarity_transform(pred_C, scale, R_opt, t_opt).flatten()
            gt_C = gt_centers[img_id]
            if np.linalg.norm(pred_transformed - gt_C) < threshold:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = (scale, R_opt, t_opt)

    maa = best_inliers / len(common_ids) if common_ids else 0.0
    return maa, best_transform

def main(submission_path, ground_truth_path):
    print(f"Loading files:\n- Submission: {submission_path}\n- Ground Truth: {ground_truth_path}")
    submission = load_submission(submission_path)
    ground_truth = load_ground_truth(ground_truth_path)

    pred_centers = get_camera_centers(submission)
    gt_centers = get_camera_centers(ground_truth)

    maa, _ = evaluate(pred_centers, gt_centers)
    print(f"✅ Mean Average Accuracy (mAA): {maa:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py submission.csv test_labels.csv")
        sys.exit(1)
    submission_file = sys.argv[1]
    gt_file = sys.argv[2]
    main(submission_file, gt_file)
# python evaluate.py submission.csv test_labels.csv
