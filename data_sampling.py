import math

import numpy as np
from scipy.spatial.distance import squareform, pdist

from Utils import item_feature_to_str, EasyRec, hierarchical_clustering

EasyRec_ReRanker = EasyRec()

def full_sampling(df):
    return df

def recent_sampling():
    pass

def relevance_sampling():
    pass

def random_sampling():
    pass

def centroid_selection_sampling():
    pass

def boundary_selection_sampling():
    pass

def centroid_selection_sampling():
    pass


def SBS_sampling(df, distance_threshold=0.5, alpha=1.1, ratio=0.6):
    print("start sbs sampling")
    positive_items=df["item_id"].tolist()
    max_index=len(positive_items)-1
    print("length of positive index:",max_index)
    pos_item_list = []
    for item_id in positive_items[:max_index]:
        item_data = df[df['item_id'] == item_id].iloc[0]
        pos_item = {
            key: value for key, value in item_data.to_dict().items()
        }
        pos_item_list.append(pos_item)

    pos_item_profile_list = []
    for pos_item in pos_item_list:
        pos_item_profile = item_feature_to_str(pos_item)
        pos_item_profile_list.append(pos_item_profile)
    print("pos_item_profile_list")
    embeddings = EasyRec_ReRanker.get_embedding(pos_item_profile_list)
    # print("embeddings:",embeddings)
    embeddings = np.array(embeddings)
    print("hierarchical_clustering")
    class2index_list, index2class = hierarchical_clustering(embeddings, distance_threshold=distance_threshold)
    print('cluster number:', len(class2index_list.keys()))
    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 10: [3, 24, 28], 3: [4, 8, 19], 9: [5], 12: [9], 11: [10, 14, 31], 2: [11, 17, 21], 6: [12, 27], 4: [13, 32], 8: [15, 29], 5: [20], 7: [22, 23]}

    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 12: [3, 24, 28], 4: [4], 11: [5], 3: [8, 19], 14: [9], 13: [10, 14, 31], 2: [11, 17, 21], 7: [12], 5: [13, 32], 10: [15, 29], 6: [20], 9: [22, 23], 8: [27]}

    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 4: [3, 10, 14, 24, 28, 31], 2: [4, 8, 11, 17, 19, 21], 3: [5, 12, 13, 15, 20, 22, 23, 27, 29, 32], 5: [9]}

    #print("class2index_list",class2index_list)
    # aa=input('pause')

    cluster_list = []
    class2centroid = {}
    for class_id, index_list in class2index_list.items():
        points = embeddings[index_list]
        cluster_list.append(points)
        centroid = np.mean(points, axis=0)
        class2centroid[class_id] = centroid

    selected_indexs_list = sampling(cluster_list, alpha=alpha, ratio=ratio)
    selected_class2index_list = {}
    for (class_label, index_list), selected_indexs in zip(class2index_list.items(), selected_indexs_list):
        selected_class2index_list[class_label] = [index_list[i] for i in selected_indexs]

    all_selected_indexs = []
    for index_list in selected_class2index_list.values():
        all_selected_indexs += index_list

    all_selected_items = []
    for index in all_selected_indexs:
        all_selected_items.append(pos_item_list[index])


    print("length of all_selected_indexs:\n",len(all_selected_indexs))
    return all_selected_indexs, all_selected_items, selected_class2index_list, class2centroid

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


def sampling_choice(df,method):
    if method=="full":
        return full_sampling(df)
    elif method=="recent":
        return recent_sampling(df)
    elif method=="SBS":
        return SBS_sampling(df)

