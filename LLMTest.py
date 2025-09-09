import os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key:", openai_api_key)

client = OpenAI()


def LLMTest():
    # print("sequence",sequence)
    response = client.responses.create(
        model="gpt-5-mini",
        input="show me a sentence to describe red"
    )
    # print("prompt:",prompt)
    # print(response.output_text)
    profileInfo = response.output_text
    print(profileInfo)

LLMTest()