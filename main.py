import time

import Constant
from DataProcess import getUID, readDataset, test_data_item
from Utils import saveProfile, exists_users
from LLMGenerate import LLMGenerate

dataset = 'Books'  # 'Books' 'xxxx' 'xxxx'
subsets = 480  # 480 10 50 200
sampling_method = "SBS"  # 'SBS' 'full' 'recent' 'relevance' 'random' 'centroid_selection' 'boundary_selection_sampling'
parameter = {"distance_threshold": 0.7, "alpha": 1.06, "ratio": 0.5}

if __name__ == '__main__':
    # 读取用户数据
    print("读取用户数据")
    df = readDataset(dataset, subsets)  # 读取数据
    ids = getUID(df)
    mode = Constant.GENERATE_MODE  # 模式

    print("对每个用户生成profile")

    # ProfilePath='./Result/{dataset}_{mode}_{subsets}/{sampling_method}/{parameter}/user_profile.json'
    if sampling_method == "SBS":
        parameter_str = '_'.join(map(str, parameter.values()))
    else:
        parameter_str = "none"
    profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=mode,
                                              sampling_method=sampling_method, parameter=parameter_str)
    user_ids = exists_users(profilePath)

    # 对每个用户生成profile
    for id in ids:
        # 如果需要更新user profile 要删除这段代码
        # if id in user_ids:
        #   continue
        candidate_list_test = test_data_item(df, id)  # 测试数据列表
        userProfileInfo = LLMGenerate(df, id, mode, sampling_method, parameter)  # 生成用户画像
        saveProfile(userProfileInfo, profilePath)  # 保存用户画像信息
