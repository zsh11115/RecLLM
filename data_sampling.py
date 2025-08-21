from Utils import item_feature_to_str


def full_sampling():
    pass

def full_sampling():
    pass

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


def analysis_behavior_sequence(pos_item_list, user_id, distance_threshold=0.5, alpha=1.1, ratio=0.6):
    if len(pos_item_list) == 0:
        print(f'user_id:{user_id} has no item')
        return
    elif len(pos_item_list) == 1:
        return [0], pos_item_list, {1: [0]}

    pos_item_profile_list = []
    for pos_item in pos_item_list:
        pos_item_profile = item_feature_to_str(pos_item)
        pos_item_profile_list.append(pos_item_profile)
    embeddings = EasyRec_ReRanker.get_embedding(pos_item_profile_list)
    # print("embeddings:",embeddings)
    embeddings = np.array(embeddings)

    class2index_list, index2class = hierarchical_clustering(embeddings, distance_threshold=distance_threshold)
    print('cluster number:', len(class2index_list.keys()))
    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 10: [3, 24, 28], 3: [4, 8, 19], 9: [5], 12: [9], 11: [10, 14, 31], 2: [11, 17, 21], 6: [12, 27], 4: [13, 32], 8: [15, 29], 5: [20], 7: [22, 23]}

    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 12: [3, 24, 28], 4: [4], 11: [5], 3: [8, 19], 14: [9], 13: [10, 14, 31], 2: [11, 17, 21], 7: [12], 5: [13, 32], 10: [15, 29], 6: [20], 9: [22, 23], 8: [27]}

    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 4: [3, 10, 14, 24, 28, 31], 2: [4, 8, 11, 17, 19, 21], 3: [5, 12, 13, 15, 20, 22, 23, 27, 29, 32], 5: [9]}

    # print("class2index_list",class2index_list)
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

    return all_selected_indexs, all_selected_items, selected_class2index_list, class2centroid