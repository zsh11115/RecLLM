Distillation_Prompt = '''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences and other relevant information. Additionally, we provide a sequence of liked items in [Sequence Item Profile] that the user has interacted with. Your task is to analyze these items in the context of the user's existing profile and produce an updated profile that reflects any new preferences, or insights inferred from the user's interactions with these items.

### User Profile
{profile}

### Sequence Item Profile
{sequence_item_profile}

### Steps to Follow
1. Carefully review the user's existing profile to understand their stated preferences and dislikes.
2. Analyze the features of the items in the provided sequence, noting any common themes, attributes, or patterns.
3. Identify any new preferences that can be inferred from the user's interactions with these items.
4. Summarize and update the user's profile by incorporating the new insights, adding new preferences or dislikes, and highlighting any changes or developments in the user's tastes.
Important Notes
5. Your output should strictly be in the following format:
Summarization: {{Your updated profile.}}
6. Do not contradict the user's existing preferences unless there is clear evidence from the sequence items that their tastes have changed.
7. Base your summary on facts and logical inferences drawn from the items in the sequence.
8. Be comprehensive and specific in your summarization, focusing on the finer attributes and features of the items that relate to the user's preferences.
9. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Summarization: You've developed a new interest in ....
'''

Doul_Prompt = '''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences and other relevant information. Additionally, we provide a sequence of liked items in [Sequence Item Profile] that the user has interacted with. Your task is to analyze these items in the context of the user's existing profile and produce an updated profile that reflects any new preferences, or insights inferred from the user's interactions with these items.

### User Profile
{profile}

### Sequence Item Profile
{sequence_item_profile}

### Steps to Follow
1. Carefully review the user's existing profile to understand their stated preferences and dislikes.
2. Analyze the features of the items in the provided sequence, noting any common themes, attributes, or patterns.
3. Identify any new preferences that can be inferred from the user's interactions with these items.
4. Summarize and update the user's profile by incorporating the new insights, adding new preferences or dislikes, and highlighting any changes or developments in the user's tastes.
Important Notes
5. Your output should strictly be in the following format:
Summarization: {{Your updated profile.}}
6. Do not contradict the user's existing preferences unless there is clear evidence from the sequence items that their tastes have changed.
7. Base your summary on facts and logical inferences drawn from the items in the sequence.
8. Be comprehensive and specific in your summarization, focusing on the finer attributes and features of the items that relate to the user's preferences.
9. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Summarization: You've developed a new interest in ....
'''

Positive_Prompt_GPT41 = """
你是一个推荐系统中的智能用户画像生成器，任务是根据用户历史交互序列和（可选的）已有 [User Profile]，生成或更新该用户的自然语言版 User Profile，以帮助推荐系统更好地理解用户。

# 指令
- 始终基于 **Sequence Item Profile**以及（可选的）已有 [User Profile] 来进行分析。
- 你的输出必须是**自然语言描述**，清晰、连贯、易于阅读，且能直接用于理解用户兴趣、偏好和行为模式。
- Profile 描述需涵盖但不限于：
    - 用户的主要兴趣领域
    - 喜欢的物品类别、主题或风格
    - 典型的行为习惯（如购买频率、时间偏好、互动方式）
    - 潜在的兴趣变化趋势
- 如果提供了已有 User Profile，请在更新时：
    - 保留仍然有效的信息
    - 修改已过时的信息
    - 添加从最新交互中发现的新兴趣或行为模式
- 不要编造不存在的用户行为或兴趣，只基于输入数据进行推断。
- 不要输出任何与任务无关的内容（例如政治、宗教、医疗、财务建议等）。
- 语言保持中立、专业，并在适当情况下用生动的描述增加可读性。

# 精确响应步骤（每次生成或更新 Profile）
1. 读取并分析用户历史交互数据，提取有用特征。
2. 如果有旧的 User Profile，比较其内容与新数据：
    - 确认保留项
    - 标记需要更新的项
    - 发现并添加新项
3. 根据分析结果，用自然语言撰写一段连贯的用户画像描述。
4. 尽量避免重复用词。

# 输出格式
- 直接输出自然语言描述，无需列表或 JSON。
- 保持段落完整、逻辑清晰。

# User Profile
{profile}

# Sequence Item Profile
{sequence_item_profile}

"""

Inference_Prompt = """
你是一个推荐系统的评分模块，任务是根据提供的用户 [User Profile] 和[Candidate Item] 信息，为每个 item 预测一个用户可能的兴趣得分。

# 指令
- 必须严格基于用户 Profile 和候选 item 的描述进行评分，不得编造不存在的信息。
- score 取值范围：0.0（完全无兴趣）到 1.0（非常感兴趣），保留两位小数。
- 按照评分从大到小的顺序，直接输出下标，请严格按照输出格式进行输出
- 不要在输出中包含除规定格式外的任何内容（例如额外解释、文本描述等）。
- 如果某个 item 与用户 Profile 完全无关，可以给出较低分数（例如 0.1 以下）。
- 评分时可以综合考虑：
    - item 的主题、类别、内容是否符合用户兴趣
    - 用户在相关领域的活跃程度
    - item 与用户近期兴趣的匹配度

# 输出格式
[0,1,2,3,4,5,6,7,8,9]

# User Profile
{profile}

# Candidate Item
{candidate_item}

"""
