import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = "Would it be a good time to visit Berlin this month?"

messages = [{
    "role": "user",
    "content": [{"text": prompt}]
}]

response = client.converse(
    modelId="amazon.nova-micro-v1:0",
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
