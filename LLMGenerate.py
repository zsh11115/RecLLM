import time

from openai import OpenAI

import Constant
from DataProcess import positive_sequence_data, doul_sequence_data, test_data_item
from prompt import Distillation_Prompt, Doul_Prompt, Positive_Prompt_GPT41, Inference_Prompt, Base_Prompt_GPT41, \
    Update_Prompt_GPT41

client = OpenAI()


def LLMGenerate(df, user_id, mode, sampling_method, parameter=None, user_profile=""):
    """
    使用LLM生成目标
    :param df: 评分数据
    :param user_id: 用户id
    :param mode: 生成模式
    :param user_profile: 用户画像
    :return: {"user_id": user_id, "output": profileInfo}
    """

    # 仅positive sequence
    if mode == Constant.GENERATE_MODE:
        sequence = positive_sequence_data(df, user_id, sampling_method, parameter)
        prompt = Base_Prompt_GPT41.format(sequence_item_profile=sequence)

    # LLM进行推荐
    if mode == Constant.REC_MODE:
        sequence = test_data_item(df, user_id)
        print(sequence)
        prompt = Inference_Prompt.format(profile=user_profile, candidate_item=sequence)

    # profile更新
    if mode == Constant.UPDATE_MODE:
        sequence = positive_sequence_data(df, user_id, 'full', parameter)
        prompt = Update_Prompt_GPT41.format(profile=user_profile, sequence_item_profile=sequence)

    # print("sequence",sequence)
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    # print("prompt:",prompt)
    # print(response.output_text)
    profileInfo = response.output_text
    return {"user_id": user_id, "output": profileInfo}
