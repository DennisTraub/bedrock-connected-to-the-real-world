import boto3
from datetime import date

client = boto3.client("bedrock-runtime", region_name="us-east-1")

today = date.today().strftime("%A %d %B %Y")
system = [{
    "text": f"Today's date is {today}. "
            f"Your are a friendly travel assistant. "
            f"Keep your responses short, with a maximum of three sentences."
}]

prompt = "Would it be a good time to visit Berlin this month?"

messages = [{
    "role": "user",
    "content": [{"text": prompt}]
}]

response = client.converse(
    modelId="amazon.nova-micro-v1:0",
    system=system,
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
