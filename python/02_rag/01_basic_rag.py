import boto3
import json
import os

from datetime import date


def load_context(file_name):
    script_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(script_path), file_name)

    with open(file_path, 'r') as file:
        return json.load(file)

client = boto3.client("bedrock-runtime", region_name="us-east-1")

system_prompt = [{
    "text": f"""
    Today's date is {date.today()}. You are a travel assistant.
    You will be given JSON data embedded in <context> tags about travel destinations and activities.
    With that information, answer the user's question, embedded in <question> tags.
    """
}]

context_data = load_context('files/travel_info.json')

context = json.dumps(context_data)

prompt = "Would it be a good time to visit Las Vegas this month?"

augmented_prompt = f"""
<context>
    {context}
</context>
<question>
    {prompt}
</question>
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
