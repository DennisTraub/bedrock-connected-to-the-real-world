import boto3
from datetime import date

client = boto3.client("bedrock-runtime", region_name="us-east-1")

system = [{
    "text": f"Today's date is {date.today()}. You are a travel assistant."
}]

messages = [
    {
      "role": "user",
      "content": [{"text": "Would it be a good time to visit Las Vegas this month?"}]
    },
    {
      "role": "assistant", 
      "content": [{"text": "Let's see... What type of activities do you like?"}]
    },
    {
      "role": "user", 
      "content": [{"text": "I like outdoor activities."}]
    }
]

response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    system=system,
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
