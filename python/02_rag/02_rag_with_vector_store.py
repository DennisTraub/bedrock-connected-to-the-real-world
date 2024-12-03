
import boto3
import chromadb

from datetime import date
from time import strftime
from chromadb.utils import embedding_functions

def get_context_from_vectordb(query):
    # Initialize ChromaDB with persistence
    chroma_client = chromadb.PersistentClient(path="./vector_db")

    # Define embedding function (default is OpenAI's text-embedding-ada-002)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Get or create the collection with the embedding function
    collection = chroma_client.get_or_create_collection(
        name="travel_info",
        embedding_function=embedding_function
    )

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=3  # Adjust number of results as needed
    )

    # Combine the retrieved chunks into a single context
    if results and results['documents']:
        return " ".join(results['documents'][0])
    return ""

# Initialize Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

date = date.today()
# date_as_text = strftime("%A %d %B %Y") // e.g. "Tuesday 03 December 2024"

system_prompt = f"""
    Today's date is {date}. You are a travel assistant.
    You will be given information in <context> tags about travel destinations and activities.
    With that information, answer the user's question, embedded in <question> tags.
    """

prompt = "Would it be a good time to visit Las Vegas this month?"

# Get relevant context from the vector DB based on the system prompt and user question
context = get_context_from_vectordb(f"{system_prompt}\n{prompt}")

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
    system=[{"text": system_prompt}],
    messages=messages
)

response_text = response["output"]["message"]["content"][0]["text"]
print(response_text)

# Helper function to initialize the vector DB with data
def initialize_vector_db():
    # Initialize with PersistentClient instead of Client
    chrome_client = chromadb.PersistentClient(path="./vector_db")

    # Define embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Create or get the collection with the embedding function
    collection = chrome_client.get_or_create_collection(
        name="travel_info",
        embedding_function=embedding_function
    )

    # Data to insert
    documents = [
        "New York - January: High 4°C/39°F, Low -3°C/27°F, Precipitation 86mm/3.4in. Cold with snow and rain likely",
        "New York - February: High 6°C/43°F, Low -2°C/28°F, Precipitation 78mm/3.1in. Cold winter weather continues, mix of snow and rain",
        "New York - March: High 10°C/50°F, Low 2°C/36°F, Precipitation 101mm/4.0in. Milder temperatures, frequent rain showers",
        "New York - April: High 16°C/61°F, Low 7°C/45°F, Precipitation 106mm/4.2in. Spring weather with regular rainfall",
        "New York - May: High 22°C/72°F, Low 13°C/55°F, Precipitation 111mm/4.4in. Warm spring temperatures, occasional thunderstorms",
        "New York - June: High 27°C/81°F, Low 18°C/64°F, Precipitation 104mm/4.1in. Warm and humid, afternoon thunderstorms possible",
        "New York - July: High 29°C/84°F, Low 21°C/70°F, Precipitation 116mm/4.6in. Hot and humid, frequent thunderstorms",
        "New York - August: High 28°C/82°F, Low 20°C/68°F, Precipitation 111mm/4.4in. Hot and humid, thunderstorms common",
        "New York - September: High 24°C/75°F, Low 16°C/61°F, Precipitation 109mm/4.3in. Warm early fall weather, occasional rain",
        "New York - October: High 18°C/64°F, Low 10°C/50°F, Precipitation 96mm/3.8in. Mild fall temperatures, moderate rainfall",
        "New York - November: High 12°C/54°F, Low 5°C/41°F, Precipitation 91mm/3.6in. Cooling temperatures, mix of rain and occasional snow",
        "New York - December: High 6°C/43°F, Low 0°C/32°F, Precipitation 94mm/3.7in. Cold with mix of rain and snow",
        "New York - Events: Broadway Shows, schedule: Various, mostly Tuesday to Sunday",
        "New York - Events: Central Park Tours, schedule: Daily",
        "New York - Events: New York Fashion Week, schedule: September and February",
        "New York - Attractions: Statue of Liberty",
        "New York - Attractions: Empire State Building",
        "New York - Attractions: Metropolitan Museum of Art",
        "New York - Attractions: Times Square",
        "Las Vegas - January: High 14°C/58°F, Low 4°C/39°F, Precipitation 13mm/0.5in. Cool and mostly clear, occasional rain",
        "Las Vegas - February: High 17°C/63°F, Low 6°C/43°F, Precipitation 18mm/0.7in. Mild days, cool nights, slightly more rain",
        "Las Vegas - March: High 21°C/70°F, Low 9°C/48°F, Precipitation 11mm/0.4in. Warming up, windy conditions common",
        "Las Vegas - April: High 26°C/78°F, Low 13°C/55°F, Precipitation 5mm/0.2in. Pleasant temperatures, very low rainfall",
        "Las Vegas - May: High 31°C/88°F, Low 18°C/64°F, Precipitation 3mm/0.1in. Getting hot, very dry conditions",
        "Las Vegas - June: High 37°C/98°F, Low 23°C/73°F, Precipitation 2mm/0.08in. Very hot and dry, clear skies",
        "Las Vegas - July: High 40°C/104°F, Low 26°C/79°F, Precipitation 11mm/0.4in. Peak summer heat, occasional monsoon storms",
        "Las Vegas - August: High 39°C/102°F, Low 26°C/78°F, Precipitation 14mm/0.5in. Very hot, highest chance of monsoon storms",
        "Las Vegas - September: High 35°C/95°F, Low 21°C/70°F, Precipitation 7mm/0.3in. Still hot but cooling down, mostly clear",
        "Las Vegas - October: High 28°C/82°F, Low 15°C/59°F, Precipitation 6mm/0.2in. Pleasant temperatures, very little rain",
        "Las Vegas - November: High 19°C/67°F, Low 8°C/47°F, Precipitation 8mm/0.3in. Cool and clear, occasional light rain",
        "Las Vegas - December: High 14°C/58°F, Low 4°C/39°F, Precipitation 13mm/0.5in. Cool winter weather, occasional rain",
        "Las Vegas - Events: Cirque du Soleil, schedule: Wednesday to Sunday",
        "Las Vegas - Events: Red Rock Canyon Tours, schedule: Daily"
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"source": "travel_info"} for _ in documents]

    # Add documents to the collection
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

if __name__ == "__main__":
    # Uncomment to initialize the vector DB with sample data
    # initialize_vector_db()
    pass