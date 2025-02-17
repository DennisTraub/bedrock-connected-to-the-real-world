import boto3
import chromadb
import os

from datetime import date
from chromadb.utils import embedding_functions

def get_context_from_vectordb(query):
    # Create a ChromaDB client
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

# Get today's date for context, e.g. "Tuesday 03 December 2024"
today = date.today().strftime("%A %d %B %Y")

system = (
    f"Today's date is {today}. You are a travel assistant."
    f"Your are a friendly travel assistant. "
    f"Keep your responses short, with a maximum of three sentences."
    f"You will be given information about travel destinations and activities embedded in <data> tags."
    f"Based on that information, answer the user's question, which is embedded in <question> tags."
)

prompt = "Would it be a good time to visit Berlin this month?"

# Get relevant context from the vector DB based on the system prompt and user question
context = get_context_from_vectordb(f"{system}\n{prompt}")

augmented_prompt = (
    f"<context>"
    f"{context}"
    f"</context>"
    f"<question>"
    f"{prompt}"
    f"</question>"
)

messages = [{"role": "user", "content": [{"text": augmented_prompt}]}]

response = client.converse(
    modelId="amazon.nova-micro-v1:0",
    system=[{"text": system}],
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
        "Las Vegas - Events: Red Rock Canyon Tours, schedule: Daily",
        "Berlin - January: High 3°C/37°F, Low -1°C/30°F, Precipitation 42mm/1.7in. Cold winter weather with occasional snow",
        "Berlin - February: High 4°C/39°F, Low -1°C/30°F, Precipitation 33mm/1.3in. Cold with mix of snow and rain",
        "Berlin - March: High 9°C/48°F, Low 2°C/36°F, Precipitation 37mm/1.5in. Gradually warming temperatures, occasional rain",
        "Berlin - April: High 14°C/57°F, Low 5°C/41°F, Precipitation 37mm/1.5in. Mild spring weather with scattered showers",
        "Berlin - May: High 19°C/66°F, Low 9°C/48°F, Precipitation 54mm/2.1in. Pleasant spring temperatures, moderate rainfall",
        "Berlin - June: High 22°C/72°F, Low 13°C/55°F, Precipitation 71mm/2.8in. Warm with occasional thunderstorms",
        "Berlin - July: High 25°C/77°F, Low 15°C/59°F, Precipitation 55mm/2.2in. Warmest month, mix of sun and rain",
        "Berlin - August: High 24°C/75°F, Low 14°C/57°F, Precipitation 58mm/2.3in. Warm summer weather continues, scattered showers",
        "Berlin - September: High 19°C/66°F, Low 11°C/52°F, Precipitation 45mm/1.8in. Pleasant early autumn temperatures",
        "Berlin - October: High 14°C/57°F, Low 7°C/45°F, Precipitation 37mm/1.5in. Cooling temperatures, moderate rainfall",
        "Berlin - November: High 8°C/46°F, Low 3°C/37°F, Precipitation 44mm/1.7in. Cool with increasing cloud cover",
        "Berlin - December: High 4°C/39°F, Low 0°C/32°F, Precipitation 55mm/2.2in. Cold winter weather with mix of rain and snow",
        "Berlin - Events: Berlinale, schedule: February",
        "Berlin - Events: Museum Island Tours, schedule: Daily",
        "Berlin - Events: Berlin Art Week, schedule: September",
        "Berlin - Attractions: Brandenburg Gate",
        "Berlin - Attractions: East Side Gallery",
        "Berlin - Attractions: Museum Island",
        "Berlin - Attractions: Reichstag Building",
        "Barcelona - January: High 14°C/57°F, Low 5°C/41°F, Precipitation 41mm/1.6in. Mild winter temperatures, occasional rain",
        "Barcelona - February: High 15°C/59°F, Low 6°C/43°F, Precipitation 29mm/1.1in. Mild days, cool nights, lower rainfall",
        "Barcelona - March: High 17°C/63°F, Low 8°C/46°F, Precipitation 40mm/1.6in. Spring begins, moderate temperatures",
        "Barcelona - April: High 19°C/66°F, Low 10°C/50°F, Precipitation 48mm/1.9in. Pleasant spring weather, occasional showers",
        "Barcelona - May: High 22°C/72°F, Low 13°C/55°F, Precipitation 47mm/1.9in. Warm days, comfortable evenings",
        "Barcelona - June: High 26°C/79°F, Low 17°C/63°F, Precipitation 29mm/1.1in. Warm and sunny, low rainfall",
        "Barcelona - July: High 29°C/84°F, Low 20°C/68°F, Precipitation 22mm/0.9in. Hot and dry, perfect beach weather",
        "Barcelona - August: High 29°C/84°F, Low 20°C/68°F, Precipitation 62mm/2.4in. Hot with occasional thunderstorms",
        "Barcelona - September: High 26°C/79°F, Low 17°C/63°F, Precipitation 81mm/3.2in. Warm, highest rainfall of the year",
        "Barcelona - October: High 22°C/72°F, Low 14°C/57°F, Precipitation 91mm/3.6in. Mild temperatures, frequent rainfall",
        "Barcelona - November: High 17°C/63°F, Low 9°C/48°F, Precipitation 58mm/2.3in. Cooling temperatures, moderate rainfall",
        "Barcelona - December: High 14°C/57°F, Low 6°C/43°F, Precipitation 40mm/1.6in. Mild winter weather, occasional rain",
        "Barcelona - Events: La Mercè Festival, schedule: September",
        "Barcelona - Events: Primavera Sound, schedule: Late May/Early June",
        "Barcelona - Events: Sagrada Familia Tours, schedule: Daily",
        "Barcelona - Attractions: Sagrada Familia",
        "Barcelona - Attractions: Park Güell",
        "Barcelona - Attractions: Casa Batlló",
        "Barcelona - Attractions: La Rambla"
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
    if not os.path.exists("./vector_db"):
        initialize_vector_db()