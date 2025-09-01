import csv
import bisect
import argparse

def calculate_rank_percentage(your_score, csv_file='leaderboard.csv'):
    scores = []
    
    # 读取 Score 列
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # 尝试从 'Score' 或 'score' 中读取
                if 'Score' in row:
                    score = float(row['Score'])
                elif 'score' in row:
                    score = float(row['score'])
                else:
                    raise KeyError("列不存在")
                scores.append(score)
            except (ValueError, KeyError) as e:
                print(f"跳过无效行: {row}, 错误: {e}")
                continue

    if not scores:
        raise ValueError("CSV 文件中没有有效的分数数据。")

    total_count = len(scores)

    # 判断是升序（越小越好）还是降序（越大越好）
    if len(scores) == 1:
        # 只有一个数据，无法判断方向，默认按降序（常见）
        is_descending = True
    else:
        is_descending = scores[0] >= scores[-1]  # 第一个 >= 最后一个 → 降序

    # === 关键：根据排序方向，确定插入位置 ===
    if is_descending:
        # 降序：高分在前，如 [95, 90, 85, 80]
        # bisect 只支持升序，所以我们将分数取负，转为升序处理
        neg_scores = [-s for s in scores]
        neg_your_score = -your_score
        rank = bisect.bisect_left(neg_scores, neg_your_score) + 1  # 排名从1开始
    else:
        # 升序：低分在前，如 [80, 85, 90, 95]
        rank = bisect.bisect_left(scores, your_score) + 1

    # 防止排名超过总人数
    rank = min(rank, total_count)

    # 计算百分比排名（越小越好）
    percentage = (rank / total_count) * 100

    return rank, total_count, percentage

def main():
    parser = argparse.ArgumentParser(description="Calculate your score on the leaderboard. The ranking percentage in CSV")
    parser.add_argument('score', type=float, help="Score")
    args = parser.parse_args()
    your_score = args.score
    rank, total, percentage = calculate_rank_percentage(your_score)

    print(f"Score: {your_score}")
    print(f"Rank: {rank} / {total}")
    print(f"Rank Percentage: {percentage:.2f}%")

if __name__ == "__main__":
    main()