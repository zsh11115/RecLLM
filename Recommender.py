import time

import requests

import Constant
from DataProcess import readDataset, getUID, test_data_item
from LLMGenerate import LLMGenerate
from Utils import get_user_profile_by_id
from evaluate import evaluate

dataset = 'Books'
subsets = 480
sampling_method = "SBS"
parameter = {"distance_threshold": 0.5, "alpha": 1.1, "ratio": 0.6}
mode = Constant.POS_MODE  # 模式


def LLM_recommender():
    """
    推荐器
    :return: score_lists
    """
    df = readDataset(dataset, subsets);
    ids = getUID(df)
    GeMode = Constant.REC_MODE

    if sampling_method == "SBS":
        parameter_str = '_'.join(map(str, parameter.values()))
    else:
        parameter_str = "none"
    profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=mode,
                                              sampling_method=sampling_method, parameter=parameter_str)
    print("对每个用户生成profile")
    score_lists = []
    # 对每个用户生成profile
    for id in ids:
        user_profile = get_user_profile_by_id(id, profilePath)
        score_list = LLMGenerate(df, id, GeMode,
                                 user_profile)  # {'user_id': 'A16Z2OECAUX8Z4', 'output': '[0,8,1,2,4,7,3,5,6,9]'}
        # print(score_list)
        score_lists.append(score_list['output'])
    print("len of user:", len(score_lists))
    return score_lists


def rank_local():
    df = readDataset(dataset, subsets);
    ids = getUID(df)
    if sampling_method == "SBS":
        parameter_str = '_'.join(map(str, parameter.values()))
    else:
        parameter_str = "none"
    profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=mode,
                                              sampling_method=sampling_method, parameter=parameter_str)
    score_lists = []
    for id in ids:
        user_profile = get_user_profile_by_id(id, profilePath)
        # 验证,用最后一个数据，加上随机取样的9个负样本，然后调用http://127.0.0.1:8001/rank, 返回的index_list
        rank_request_data = {}
        rank_request_data['user'] = {'user_id': id, 'user_persona': user_profile, 'pos_user_persona': user_profile,
                                     'neg_user_persona': 'Currently Unknown'}
        rank_request_data['items'] = test_data_item(df,id)
        rank_request_data['model_name'] = 'EasyRec'
        rank_request_data['api_key'] = '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'
        try:
            rank_response = requests.post('http://127.0.0.1:8001/rank_local', json=rank_request_data).json()
        except Exception as e:
            print(e)
            print(rank_request_data)
            aa = input('error happened, pause')
        index_list = rank_response['index_list']
        print(index_list)
        score_lists.append(index_list)
    return score_lists

if __name__ == '__main__':
    score_lists = LLM_recommender()
    #score_lists=rank_local()
    evaluate(score_lists)

"""
len(user):100
K=1 | Hit@1=0.4700 | NDCG@1=0.4700 | MRR@1=0.4700
K=5 | Hit@5=0.9800 | NDCG@5=0.7436 | MRR@5=0.6650
K=10 | Hit@10=1.0000 | NDCG@10=0.7498 | MRR@10=0.6674

rank_local:
K=1 | Hit@1=0.3400 | NDCG@1=0.3400 | MRR@1=0.3400
K=5 | Hit@5=0.7400 | NDCG@5=0.5516 | MRR@5=0.4890
K=10 | Hit@10=1.0000 | NDCG@10=0.6345 | MRR@10=0.5226
"""
