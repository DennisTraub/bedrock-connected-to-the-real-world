import boto3
import json
from datetime import date

def load_context(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

client = boto3.client("bedrock-runtime", region_name="us-east-1")
system_prompt = [{
    "text": f"""
    Today's date is {date.today()}. You are a travel assistant.
    You will be given a context about travel destinations and activities.
    Answer the user's questions based on the context.
    You MUST follow the rules below:
    - If the context doesn't contain the answer, say that you don't know the answer.
    - If the question is not related to travel, say that you don't know the answer.
    """
}]

context_data = load_context('../data/travel_info.json')

context = json.dumps(context_data)

prompt = "Would it be a good time to visit Las Vegas this month?"

augmented_prompt = f"""
<CONTEXT>{context}</CONTEXT>
<USER_QUESTION>{prompt}</USER_QUESTION>
"""

messages = [{
    "role": "user", 
    "content": [{"text": augmented_prompt}]   
}]

response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    system=system_prompt,
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
