import boto3
from datetime import date

client = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = "Would it be a good time to visit Las Vegas this month?"

messages = [{
    "role": "user",
    "content": [{"text": prompt}]
}]

system_prompt = f"Today's date is {date.today()}. You are a travel assistant."

response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    system=[{"text": system_prompt}],
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
