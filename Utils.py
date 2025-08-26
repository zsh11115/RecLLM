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


def hierarchical_clustering(embeddings, distance_threshold=0.3):
    Z = linkage(embeddings, method='ward', metric='euclidean')

    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    original_indices = np.arange(len(embeddings))

    class2index_list = defaultdict(list)
    index2class = {}

    for original_idx, label in zip(original_indices, labels):
        class2index_list[label].append(original_idx)
        index2class[original_idx] = label

    return dict(class2index_list), index2class