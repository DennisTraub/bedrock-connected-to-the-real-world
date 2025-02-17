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

# Get today's date for context, e.g. "Tuesday 03 December 2024"
today = date.today().strftime("%A %d %B %Y")

# Define the system prompt with instructions for handling the embedded data
system = [{
    "text": f"Today's date is {today}. You are a travel assistant."
            f"Your are a friendly travel assistant. "
            f"Keep your responses short, with a maximum of three sentences."
            f"You will be given information about travel destinations and activities embedded in <data> tags."
            f"Based on that information, answer the user's question, which is embedded in <question> tags."
}]

context_data = load_context("files/travel_info.json")

context = json.dumps(context_data)

prompt = "Would it be a good time to visit Berlin this month?"

augmented_prompt = (
    f"<context>"
    f"{context}"
    f"</context>"
    f"<question>"
    f"{prompt}"
    f"</question>"
)

messages = [{
    "role": "user", 
    "content": [{"text": augmented_prompt}]   
}]

response = client.converse(
    modelId="amazon.nova-micro-v1:0",
    system=system,
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]

print(response_text)
