Base_Prompt_GPT41 = """
### Task
we provide a sequence of items in [Sequence Item Profile] that the user has interacted with. Your task is to analyze 
these items and produce an updated profile that reflects any new preferences, or insights inferred from the user's 
interactions with these items.

### Sequence Item Profile
{sequence_item_profile}

### Steps to Follow
1. Analyze the features of the items in the provided sequence, noting any common themes, attributes, or patterns.
2. Identify any  preferences that can be inferred from the user's interactions with these items.
3. Summarize the user's profile by incorporating the new insights, dislikes in the user's tastes.
Important Notes
4. Your output should strictly be in the following format:
Summarization: {{Your updated profile.}}
5. Base your summary on facts and logical inferences drawn from the items in the sequence.
6. Be comprehensive and specific in your summarization, focusing on the finer attributes and features of the items that 
relate to the user's preferences.
7. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Summarization: You've developed a new interest in ....
"""

Update_Prompt_GPT41 = """
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences and other relevant 
information. Additionally, we provide a sequence of items in [Sequence Item Profile] that the user has interacted with. 
Your task is to analyze these items in the context of the user's existing profile and produce an updated profile that 
reflects any new preferences, or insights inferred from the user's interactions with these items.

### User Profile
{profile}

### Sequence Item Profile
{sequence_item_profile}

### Steps to Follow
1. Carefully review the user's existing profile to understand their stated preferences and dislikes.
2. Analyze the features of the items in the provided sequence, noting any common themes, attributes, or patterns.
3. Identify any new preferences that can be inferred from the user's interactions with these items.
4. Summarize and update the user's profile by incorporating the new insights, adding new preferences or dislikes, and 
highlighting any changes or developments in the user's tastes.
Important Notes
5. Your output should strictly be in the following format:
Summarization: {{Your updated profile.}}
6. Do not contradict the user's existing preferences unless there is clear evidence from the sequence items that their 
tastes have changed.
7. Base your summary on facts and logical inferences drawn from the items in the sequence.
8. Be comprehensive and specific in your summarization, focusing on the finer attributes and features of the items that 
relate to the user's preferences.
9. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Summarization: You've developed a new interest in ....
"""