import math
import os
import json

import numpy as np
import requests
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict

def saveProfile(content, filePath):
    # 获取 user_id 和 profile
    user_id = content["user_id"]
    profile = content["output"]

    if user_id is None or profile is None:
        raise ValueError("content 必须包含 'user_id' 和 'profile' 字段")

        # 自动创建目录
    os.makedirs(os.path.dirname(filePath), exist_ok=True)

    # 构建 {user_id: {profile: ...}} 格式
    new_entry = {user_id: {"profile": profile}}

    # 如果文件存在就读取
    if os.path.exists(filePath):
        with open(filePath, 'r') as f:
            try:
                existed_user_personas_dict = json.load(f)
            except json.JSONDecodeError:
                existed_user_personas_dict = {}
    else:
        existed_user_personas_dict = {}

    # 更新或添加内容
    if user_id in existed_user_personas_dict:
        existed_user_personas_dict[user_id]['profile'] = profile
    else:
        existed_user_personas_dict.update(new_entry)

    # 写回文件
    with open(filePath, 'w') as f:
        json.dump(existed_user_personas_dict, f, indent=4, ensure_ascii=False)


def exists_users(filePath):
    # 如果文件存在就读取
    if os.path.exists(filePath):
        with open(filePath, 'r') as f:
            try:
                existed_user_personas_dict = json.load(f)
            except json.JSONDecodeError:
                existed_user_personas_dict = {}
    else:
        existed_user_personas_dict = {}

    user_ids=[]
    for record in existed_user_personas_dict:
        user_ids.append(record)
    return user_ids

def get_user_profile_by_id(user_id, filePath):
    # 如果文件存在就读取
    if os.path.exists(filePath):
        with open(filePath, 'r') as f:
            try:
                existed_user_personas_dict = json.load(f)
            except json.JSONDecodeError:
                existed_user_personas_dict = {}
    else:
        existed_user_personas_dict = {}
    profile = existed_user_personas_dict[user_id]['profile']
    # print(profile)
    return profile


def item_feature_to_str(item_feature):
    """
    Convert item feature to string.
    :param item_feature: The feature of an item, which is a dictionary.
    """
    assert isinstance(item_feature, dict), f"item_feature should be a dictionary, but got {type(item_feature)}"
    feature_str = ""
    for key, value in item_feature.items():
        if 'id' in key:
            continue
        else:
            feature_str += f"{key}:{value}\n"
    return feature_str


class EasyRec:
    def __init__(self, url='http://localhost:8500'):
        self.url = url

    def get_embedding(self, documents):
        response = requests.post(f"{self.url}/get_embedding", json={"documents": documents})
        #print("Status:", response.status_code)
        #print("Response:", response.text)
        embeddings = response.json()['embeddings']
        return embeddings

    def predict(self, query, documents):
        response = requests.post(f"{self.url}/compute_scores", json={"query": query, "documents": documents})
        scores = response.json()['scores']
        return scores


def hierarchical_clustering(embeddings, distance_threshold=0.5):
    Z = linkage(embeddings, method='ward', metric='euclidean')

    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    original_indices = np.arange(len(embeddings))

    class2index_list = defaultdict(list)
    index2class = {}

    for original_idx, label in zip(original_indices, labels):
        class2index_list[label].append(original_idx)
        index2class[original_idx] = label

    return dict(class2index_list), index2class



from scipy.spatial.distance import squareform, pdist
def sampling(cluster_list, alpha=1.1, ratio=0.6):
    sum_size = sum([len(cluster) for cluster in cluster_list])
    budget = math.ceil(sum_size * ratio)
    cluster_size_list = [len(cluster_list[i]) for i in range(len(cluster_list))]
    allocation = get_allocation(cluster_size_list, budget)

    selected_indexs_list = []
    for i in range(len(cluster_list)):
        cluster = cluster_list[i]
        selected_points, selected_indexs, selected_scores = select_samples(cluster, alpha, allocation[i])
        selected_indexs_list.append(selected_indexs)

    return selected_indexs_list


def get_allocation(cluster_size_list, budget):
    # The budget must be at least equal to or greater than the number of clusters, so that each cluster can be allocated at least one
    if budget < len(cluster_size_list):
        budget = len(cluster_size_list)

    # Convert to numpy array for easier handling
    cluster_size_list = np.array(cluster_size_list)

    # Sort cluster sizes in ascending order and get their sorted indices
    cluster_size_list_index = np.argsort(cluster_size_list)
    sorted_cluster_size_list = cluster_size_list[cluster_size_list_index]

    # Initialize allocation list
    allocation_list = []

    # Distribute the budget
    for i in range(len(sorted_cluster_size_list)):
        unallocated_strata = len(sorted_cluster_size_list) - len(allocation_list)
        avg = budget // unallocated_strata
        cluster_size = sorted_cluster_size_list[i]

        # Allocate budget
        if cluster_size <= avg:
            allocation_list.append(cluster_size)
        else:
            allocation_list.append(avg)

        # Update the remaining budget
        budget -= allocation_list[-1]

    if budget > 0:
        for i in range(budget):
            allocation_list[-(i + 1)] += 1

    # Restore the allocation list to the original order
    final_allocation_list = [0] * len(cluster_size_list)
    for i, index in enumerate(cluster_size_list_index):
        final_allocation_list[index] = allocation_list[i]

    return final_allocation_list

def select_samples(points, alpha=1.1, num_samples=10):
    center = points.mean(axis=0)
    size = len(points)
    ratio = num_samples / size

    def cal_aim(c, point, selected_points):
        w_p = alpha ** (-10)
        w_d = 1 - w_p

        distance = np.linalg.norm(point - center)
        easy = w_p * 1 / (1 + distance)

        if len(selected_points) == 0:
            if_selected_points = np.array([point])
        else:
            if_selected_points = np.concatenate((selected_points, [point]), axis=0)

        # Calculate the distance sum of two points in if_selected_points
        distance_matrix = squareform(pdist(if_selected_points, metric='euclidean'))
        distance_matrix = np.triu(distance_matrix, k=1)
        distance_avg = np.sum(distance_matrix) / len(if_selected_points)
        diversity = w_d * distance_avg
        score = easy + diversity
        return score, easy, diversity

    selected_points = []
    selected_indexs = []
    selected_scores = []
    for i in range(num_samples):
        scores = [cal_aim(c, point, selected_points) for c, point in enumerate(points)]
        sum_score = [score[0] for score in scores]
        easy_score = [score[1] for score in scores]
        diversity_score = [score[2] for score in scores]

        valid_indices = [j for j in range(len(sum_score)) if j not in selected_indexs]
        max_valid_index = np.argmax(np.take(sum_score, valid_indices))
        max_index = valid_indices[max_valid_index]

        selected_points.append(points[max_index])
        selected_indexs.append(max_index)
        selected_scores.append(scores[max_index])

    return np.array(selected_points), selected_indexs, selected_scores