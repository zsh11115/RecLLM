import numpy as np

from Utils import item_feature_to_str, EasyRec, hierarchical_clustering, sampling

EasyRec_ReRanker = EasyRec()

def sampling_choice(df,method,parameter):
    if method=="full":
        return full_sampling(df)
    elif method=="recent":
        return recent_sampling(df)
    elif method=="relevance":
        return relevance_sampling(df)
    elif method=="random":
        return random_sampling(df)
    elif method=="centroid_selection":
        return centroid_selection_sampling(df)
    elif method=="boundary_selection":
        return boundary_selection_sampling(df)
    elif method=="SBS":
        return SBS_sampling(df,parameter)


def full_sampling(df):
    """
    全部sequence
    :param df:
    :return:
    """
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



def SBS_sampling(df, parameter):
    distance_threshold=parameter["distance_threshold"]
    alpha=parameter["alpha"]
    ratio=parameter["ratio"]
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
    print("selected_index",all_selected_indexs)
    # return all_selected_indexs, all_selected_items, selected_class2index_list, class2centroid
    return all_selected_items




