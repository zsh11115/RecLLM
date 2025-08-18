from Constant import DataSetsFile
import pandas as pd

def readDataset(dataset, subsets):
    dataFile = DataSetsFile.format(dataset=dataset, subsets=subsets)
    df = pd.read_csv(dataFile)
    return df


# 用户id数据
def getUID(df):
    user_ids = df["user_id"].unique().tolist()
    return user_ids


def getItemId(df):
    item_ids = df["item_id"].unique().tolist()
    return item_ids


def positive_sequence_data(df, user_id):
    sequence = []
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')
    positive_data = user_data[user_data['rating'] >= 1]
    negitive_data = user_data[user_data['rating'] < 1]
    lenOfData = len(positive_data)
    # print("user_id:", user_id, "length of data:", lenOfData)
    # print("positive data:",positive_data)
    return {"pos": positive_data}


def doul_sequence_data(df, user_id):
    sequence = []
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')
    positive_data = user_data[user_data['rating'] >= 1]
    negitive_data = user_data[user_data['rating'] < 1]
    lenOfData = len(positive_data)
    # print("user_id:", user_id, "length of data:", lenOfData)
    # print("positive data:",positive_data)
    return {"pos": positive_data, "neg": negitive_data}


def test_data_item(df, user_id):
    """unique_items=getItemId(df)
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')
    positive_data = user_data[user_data['rating'] >= 1]
    negative_data = user_data[user_data['rating'] < 1]
    print("negative_data:", user_data)
    negative_samples = negative_data.sample(9, random_state=42)
    print("negative_samples:",negative_samples)"""

    all_data = df.sort_values(by='timestamp')
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')

    # 正样本：取最后一条，去掉 rating/timestamp，转 dict
    positive_data = user_data[user_data['rating'] >= 1]
    last_positive_dict = positive_data.iloc[-1].drop(['rating', 'timestamp']).to_dict()
    pos_item_ids = set(positive_data['item_id'])

    # 负样本：随机取 9 条，去掉 rating/timestamp，转 dict
    negative_data = all_data[~all_data['item_id'].isin(pos_item_ids)]
    negative_dicts = (
        negative_data
        .drop(['rating', 'timestamp'], axis=1)
        .sample(9, random_state=42)
        .to_dict(orient='records')
    )
    test_list=[last_positive_dict] + negative_dicts
    #print(test_list)  #[{positive},{negative},{negative},{negative},{negative},{negative},{negative},{negative},{negative},{negative}]
    return test_list
