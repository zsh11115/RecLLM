from Constant import DataSetsFile
import pandas as pd

from data_sampling import sampling_choice


def readDataset(dataset, subsets):
    """
    读取数据集数据
    :param dataset:
    :param subsets:
    :return: df->dataframe
    """
    dataFile = DataSetsFile.format(dataset=dataset, subsets=subsets)
    df = pd.read_csv(dataFile)
    return df


# 用户id数据
def getUID(df):
    """
    获取用户id
    :param df:
    :return: 用户id列表->list
    """
    user_ids = df["user_id"].unique().tolist()
    return user_ids


def getItemId(df):
    """
    获取物品id
    :param df:
    :return: 物品id列表->list
    """
    item_ids = df["item_id"].unique().tolist()
    return item_ids


def positive_sequence_data(df, user_id, sampling_method, parameter):
    """
    仅考虑positive的行为序列
    :param parameter: 采样参数
    :param sampling_method: 采样方法
    :param df:
    :param user_id:
    :return: {"pos": positive_data}
    """
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')

    positive_data = user_data[user_data['rating'] >= 1]
    positive_data = positive_data.drop(['rating', 'timestamp'], axis=1)

    # 对序列进行采样
    sample_method = sampling_method
    all_selected_items = sampling_choice(positive_data, sample_method, parameter)

    return {"pos": all_selected_items}


def doul_sequence_data(df, user_id, sampling_method, parameter):
    """
    考虑positive和negative的行为序列
    :param df:
    :param user_id:
    :return: {"pos": positive_data, "neg": negative_data}
    """
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')
    positive_data = user_data[user_data['rating'] >= 1]
    negative_data = user_data[user_data['rating'] < 1]
    return {"pos": positive_data, "neg": negative_data}


def test_data_item(df, user_id):
    """
    测试数据列表
    :param df:
    :param user_id:
    :return:[{},{},{},...,{}]
    """

    all_data = df.sort_values(by='timestamp')
    user_data = df[df['user_id'] == user_id].sort_values(by='timestamp')

    # 正样本：取最后一条，去掉 rating/timestamp，转 dict
    positive_data = user_data[user_data['rating'] >= 1]
    last_positive_dict = positive_data.iloc[-1].drop(['user_id', 'rating', 'timestamp']).to_dict()
    pos_item_ids = set(positive_data['item_id'])

    # 负样本：随机取 9 条，去掉 rating/timestamp，转 dict
    negative_data = all_data[~all_data['item_id'].isin(pos_item_ids)]
    negative_dicts = (
        negative_data
        .drop(['user_id', 'rating', 'timestamp'], axis=1)
        .sample(9, random_state=42)
        .to_dict(orient='records')
    )
    test_list = [last_positive_dict] + negative_dicts

    return test_list
