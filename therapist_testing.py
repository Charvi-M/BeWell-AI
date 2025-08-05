#from rag_pipeline import multiagent_chain
from FAISS_rag_pipeline import multiagent_chain

print("🧠 Mental Health Therapist (AI) - FIXED VERSION")
print("Type 'exit' to quit.")
print("Tip: If you're financially struggling, try 'I can't afford therapy'.")

# Simulated user profile (you can customize this)
user_profile = {
    "name": "Test User",
    "country": "India",
    "financial": "low",
    "diagnosis": "anxiety",
    "is_minor": False
}

print(f"\n👤 User Profile: {user_profile['country']}, Financial: {user_profile['financial']}")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "exit":
        print("Therapist: Take care. You're not alone.")
        break

    try:
        result = multiagent_chain(user_input, user_profile)
        agent = result["agent"]
        response = result["response"]

        print(f"\n{agent}: {response}")
    except Exception as e:
        print(f"[System Error] {str(e)}")
        import traceback
        traceback.print_exc()