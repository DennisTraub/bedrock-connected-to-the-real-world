import boto3
from datetime import date

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Get today's date for context, e.g. "Tuesday 03 December 2024"
today = date.today().strftime("%A %d %B %Y")

# Define the system prompt
system = [{
    "text": f"Today's date is {today}. "
            f"Your are a friendly travel assistant. "
            f"Keep your responses short, with a maximum of three sentences."
}]

messages = [
    {
      "role": "user",
      "content": [{"text": "Would it be a good time to visit Berlin this month?"}]
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
    modelId="amazon.nova-micro-v1:0",
    system=system,
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
