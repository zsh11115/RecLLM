import numpy as np
import ast


#  统计所有用户的评估参数，进行均值计算
def evaluate(raw_data):
    print(raw_data)
    recommendations = [ast.literal_eval(item) for item in raw_data]
    topk = [1, 5, 10]
    for k in topk:
        hit = hit_at_k(recommendations, k)
        ndcg = ndcg_at_k(recommendations, k)
        mrr = mrr_at_k(recommendations, k)
        print(f"K={k} | Hit@{k}={hit:.4f} | NDCG@{k}={ndcg:.4f} | MRR@{k}={mrr:.4f}")


import numpy as np


def hit_at_k(recommendations, k):
    hits = [1 if 0 in rec[:k] else 0 for rec in recommendations]
    return np.mean(hits)


def ndcg_at_k(recommendations, k):
    ndcg_scores = []
    for rec in recommendations:
        dcg = 0
        for i in range(k):
            if rec[i] == 0:
                dcg = 1 / np.log2(i + 2)
                break
        idcg = 1 / np.log2(2)  # 因为只有一个正样本
        ndcg_scores.append(dcg / idcg)
    return np.mean(ndcg_scores)


def mrr_at_k(recommendations, k):
    reciprocal_ranks = []
    for rec in recommendations:
        for i in range(min(k, len(rec))):
            if rec[i] == 0:
                reciprocal_ranks.append(1 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)
