import time

import Constant
from DataProcess import getUID, readDataset, test_data_item
from Utils import saveProfile
from LLMGenerate import LLMGenerate

dataset = 'Books'
subsets = 480

if __name__ == '__main__':
    # 读取用户数据
    print("读取用户数据")
    df = readDataset(dataset, subsets);
    ids = getUID(df)
    mode = Constant.DOUL_MODE
    print("对每个用户生成profile")
    # 对每个用户生成profile
    for id in ids:
        candidate_list_test=test_data_item(df, id)
        #time.sleep(30)
        userProfileInfo=LLMGenerate(df,id,mode)
        print("userProfileInfo:",userProfileInfo)
        profilePath=Constant.ProfilePath.format(dataset=dataset,subsets=subsets,mode=mode)
        saveProfile(userProfileInfo,profilePath)
