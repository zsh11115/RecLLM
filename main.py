import time

import Constant
from DataProcess import getUID, readDataset, test_data_item
from Utils import saveProfile
from LLMGenerate import LLMGenerate

dataset = 'Books'
subsets = 480
sampling_method="SBS"
parameter={"distance_threshold":0.5,"alpha":1.1,"ratio":0.6}

if __name__ == '__main__':
    # 读取用户数据
    print("读取用户数据")
    df = readDataset(dataset, subsets)  # 读取数据
    ids = getUID(df)
    mode = Constant.POS_MODE  # 模式
    print("对每个用户生成profile")
    # 对每个用户生成profile
    print("length of id:",len(ids))
    for id in ids:
        candidate_list_test = test_data_item(df, id)  # 测试数据列表
        # time.sleep(30)
        userProfileInfo = LLMGenerate(df, id, mode,sampling_method,parameter)  # 生成用户画像
        print("userProfileInfo:", userProfileInfo)

        # ProfilePath='./Result/{dataset}_{mode}_{subsets}/{sampling_method}/{parameter}/user_profile.json'
        if sampling_method == "SBS":
            parameter_str='_'.join(map(str, parameter.values()))
        else:
            parameter_str="none"
        profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=mode,sampling_method=sampling_method,parameter=parameter_str)



        saveProfile(userProfileInfo, profilePath)  # 保存用户画像信息
