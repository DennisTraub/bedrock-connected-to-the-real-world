import boto3

from time import strftime


def get_weather_tool_spec():
    """
    Returns the JSON Schema specification for the get_current_weather tool. The tool specification defines the input schema and describes the tool's functionality.
    For more information, see https://json-schema.org/understanding-json-schema/reference.

    :return: The tool specification for the get_current_weather tool.
    """
    return {
        "toolSpec": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get weather for",
                        }
                    },
                    "required": ["city"],
                }
            },
        }
    }

def weather_tool(city):
    weather_data = {
        "new_york": {"temperature": 40, "condition": "Partly cloudy"},
        "las_vegas": {"temperature": 71, "condition": "Sunny"}
    }

    return weather_data.get(city.lower().replace(" ", "_"))

def process_response(follow_up_response, depth=0, max_depth=3):
    # Check recursion depth first
    if depth >= max_depth:
        print(f"Maximum recursion depth ({max_depth}) reached")
        return

    # Append the model's response to the conversation
    messages.append(follow_up_response["output"]["message"])

    # Check each individual content block of the response
    for content_block in follow_up_response["output"]["message"]["content"]:

        # Display text responses in the console
        if "text" in content_block:
            response_text = follow_up_response["output"]["message"]["content"][0]["text"]
            print(f"\n{response_text}")

        # Process tool use requests
        elif "toolUse" in content_block:
            tool_use_request = content_block["toolUse"]
            if tool_use_request["name"] == "get_current_weather":
                tool_use_id = tool_use_request["toolUseId"]
                city = tool_use_request["input"]["city"]
                weather_info = weather_tool(city)

                # Append the tool's response to the conversation
                messages.append({
                    "role": "user",
                    "content": [
                        {"toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": weather_info}],
                        }}
                    ]
                })

                # Send the tool's response back to the model
                follow_up_response = client.converse(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    toolConfig=tool_config,
                    system=system_prompt,
                    messages=messages
                )

                process_response(follow_up_response, depth + 1, max_depth)

client = boto3.client("bedrock-runtime", region_name="us-east-1")

date_as_text = strftime("%A %d %B %Y")

system_prompt = [{
    "text": f"""
    Today's date is {date_as_text}. You are a travel assistant.
    You also have access to a tool get_current_weather.
    With this in mind, answer the user's questions.
    You MUST follow the rules below:
    - ALWAYS use the get_current_weather to get current weather information.
    - Don't rely on anything else for weather information.
    - Don't make up weather information.
    - If the tool doesn't return the weather, say that you don't know the answer.
    - If the question is not related to travel, say that you don't know the answer.
    """
}]

prompt = "Would it be a good time to visit Las Vegas this month?"

messages = [{
    "role": "user", 
    "content": [{"text": prompt}]   
}]

tool_config = {"tools": [get_weather_tool_spec()]}

response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    toolConfig=tool_config,
    system=system_prompt,
    messages=messages
)

process_response(response)
