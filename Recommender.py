import time

import Constant
from DataProcess import readDataset, getUID, test_data_item
from LLMGenerate import LLMGenerate
from Utils import get_user_profile_by_id
from evaluate import evaluate

dataset = 'Books'
subsets = 480


def recommender():
    df = readDataset(dataset, subsets);
    ids = getUID(df)
    GeMode = Constant.REC_MODE
    FileMode=Constant.POS_MODE
    profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=FileMode)
    print("对每个用户生成profile")
    score_lists=[]
    # 对每个用户生成profile
    for id in ids:
        user_profile = get_user_profile_by_id(id, profilePath)
        score_list = LLMGenerate(df, id, GeMode, user_profile) # {'user_id': 'A16Z2OECAUX8Z4', 'output': '[0,8,1,2,4,7,3,5,6,9]'}
        #print(score_list)
        score_lists.append(score_list['output'])
    print("len of user:",len(score_lists))
    return score_lists


if __name__ == '__main__':
    score_lists=recommender()
    evaluate(score_lists)



"""
len(user):100
K=1 | Hit@1=0.4700 | NDCG@1=0.4700 | MRR@1=0.4700
K=5 | Hit@5=0.9800 | NDCG@5=0.7436 | MRR@5=0.6650
K=10 | Hit@10=1.0000 | NDCG@10=0.7498 | MRR@10=0.6674
"""