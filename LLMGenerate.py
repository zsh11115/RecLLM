import time

from openai import OpenAI

import Constant
from DataProcess import positive_sequence_data, doul_sequence_data, test_data_item
from prompt import Distillation_Prompt, Doul_Prompt, Positive_Prompt_GPT41, Inference_Prompt

client = OpenAI()


def LLMGenerate(df, user_id, mode, sampling_method,parameter,user_profile=""):
    """
    使用LLM生成目标
    :param df:
    :param user_id:
    :param mode:
    :param user_profile:
    :return: {"user_id": user_id, "output": profileInfo}
    """
    print("mode:", mode)
    if mode == Constant.POS_MODE:
        sequence = positive_sequence_data(df, user_id,sampling_method,parameter)
        prompt = Positive_Prompt_GPT41.format(profile=user_profile, sequence_item_profile=sequence)

    if mode == Constant.DOUL_MODE:
        sequence = doul_sequence_data(df, user_id,sampling_method,parameter)
        prompt = Doul_Prompt.format(profile=user_profile, sequence_item_profile=sequence)

    if mode==Constant.REC_MODE:
        sequence = test_data_item(df,user_id)
        print(sequence)
        prompt = Inference_Prompt.format(profile=user_profile, candidate_item=sequence)
    #print("sequence",sequence)
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    # print("prompt:",prompt)
    # print(response.output_text)
    profileInfo = response.output_text
    return {"user_id": user_id, "output": profileInfo}
