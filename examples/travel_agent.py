import openai
import psycopg2
import pinecone
from sentence_transformers import SentenceTransformer
import requests

# Initialize OpenAI
openai.api_key = "your_openai_api_key"

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")
index_name = "trip-planner-interests"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# PostgreSQL connection
def connect_to_postgres():
    return psycopg2.connect(
        dbname="trip_planner",
        user="your_postgres_user",
        password="your_postgres_password",
        host="localhost",
        port="5432"
    )

# Function to upsert user profile into PostgreSQL
def upsert_user_profile(user_id, profile_data):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_profiles (user_id, name, email, preferences)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE
        SET name = EXCLUDED.name,
            email = EXCLUDED.email,
            preferences = EXCLUDED.preferences;
    """, (user_id, profile_data['name'], profile_data['email'], profile_data['preferences']))
    conn.commit()
    cursor.close()
    conn.close()

# Function to query user profile from PostgreSQL
def query_user_profile(user_id):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s;", (user_id,))
    user_profile = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_profile:
        return {
            "user_id": user_profile[0],
            "name": user_profile[1],
            "email": user_profile[2],
            "preferences": user_profile[3]
        }
    return None

# Function to upsert user interests/preferences into Pinecone
def upsert_user_interests(user_id, interests_data):
    embedding = embedding_model.encode(interests_data)
    index.upsert([(user_id, embedding, {"interests": interests_data})])

# Function to query user interests/preferences from Pinecone
def query_user_interests(user_id, query_text, top_k=1):
    query_embedding = embedding_model.encode(query_text)
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return results

# Function to dynamically generate steps with context
def get_trip_planning_steps(context):
    prompt = f"""
    I want to plan a trip. Here is some context about the user:
    {context}
    Please outline the steps for planning a trip, including selecting a destination, finding flights, and other necessary actions.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to suggest destinations using context
def suggest_destinations(context, preferences):
    prompt = f"""
    Based on the following context:
    {context}
    And the user's preferences:
    - Destination type: {preferences['type']}
    - Budget: {preferences['budget']}
    - Interests: {preferences['interests']}
    
    Suggest three destinations along with reasons to visit each.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Function to fetch flights
def fetch_flights(origin, destination, date):
    # Dummy API URL (replace with a real API like Skyscanner or Amadeus)
    api_url = "https://api.example.com/flights"
    headers = {"Authorization": "Bearer your_api_token"}
    params = {
        "origin": origin,
        "destination": destination,
        "date": date,
    }
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Unable to fetch flights"}

# Function to book a flight
def book_flight(flight_id, passenger_details):
    # Dummy API URL (replace with a real API)
    api_url = "https://api.example.com/bookings"
    headers = {"Authorization": "Bearer your_api_token"}
    data = {
        "flight_id": flight_id,
        "passenger_details": passenger_details,
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 201:
        return response.json()
    else:
        return {"error": "Unable to book the flight"}

# Main program
if __name__ == "__main__":
    print("Welcome to the AI Trip Planner with Contextual Insights!")

    # Step 1: Save user profile in PostgreSQL
    user_id = "user123"
    user_profile_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "preferences": "Prefers luxury travel and cultural experiences."
    }
    upsert_user_profile(user_id, user_profile_data)
    print("User profile saved to PostgreSQL.")

    # Step 2: Save user interests/preferences in Pinecone
    user_interests = "Likes beaches, history, and culinary tours."
    upsert_user_interests(user_id, user_interests)
    print("User interests saved to Pinecone.")

    # Step 3: Retrieve context from PostgreSQL and Pinecone
    print("\nRetrieving user context...")
    profile_context = query_user_profile(user_id)
    interests_context = query_user_interests(user_id, "Find a suitable travel destination")
    if interests_context['matches']:
        interests_context = interests_context['matches'][0]['metadata']['interests']
    else:
        interests_context = "No specific interests found."

    # Combine contexts
    full_context = f"Profile: {profile_context['preferences']}\nInterests: {interests_context}"

    # Step 4: Generate trip planning steps
    print("\nGenerating trip planning steps...")
    steps = get_trip_planning_steps(full_context)
    print(f"\nSteps for planning your trip:\n{steps}")

    # Execute steps dynamically
    if "destination" in steps.lower():
        # Step: Get preferences
        preferences = {
            "type": input("\nWhat type of destination do you prefer (e.g., beach, city, adventure)? "),
            "budget": input("What is your budget (e.g., $1000)? "),
            "interests": input("What are your interests (e.g., hiking, shopping)? "),
        }

        # Step: Get destination suggestions
        print("\nFinding destinations...")
        destinations = get_destination_suggestions(preferences)
        print(f"\nHere are some suggestions:\n{destinations}")
        selected_destination = input("\nWhich destination would you like to visit? ")

    if "flights" in steps.lower():
        # Step: Fetch flight options
        print("\nFetching flight options...")
        flights = fetch_flights("JFK", selected_destination, "2025-02-15")  # Replace JFK with your origin
        if "error" in flights:
            print("Error fetching flights:", flights["error"])
        else:
            print("\nAvailable flights:")
            for flight in flights["results"]:
                print(f"Flight ID: {flight['id']}, Price: {flight['price']}, Duration: {flight['duration']}")

            # Step: Book a flight
            flight_id = input("\nEnter the Flight ID to book: ")
            passenger_details = {
                "name": input("Enter your full name: "),
                "email": input("Enter your email: "),
                "phone": input("Enter your phone number: "),
            }
            booking_response = book_flight(flight_id, passenger_details)
            if "error" in booking_response:
                print("Error booking flight:", booking_response["error"])
            else:
                print("\nBooking confirmed! Here are the details:")
                print(booking_response)
