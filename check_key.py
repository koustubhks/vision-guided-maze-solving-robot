import os
from google import genai
from google.genai.errors import APIError

def check_api_key_status():
    """
    Attempts a simple API call (listing models) to verify key activation.
    Requires GEMINI_API_KEY to be set in the environment.
    """
    print("--- Starting API Key Check ---")

    # 1. Check for the Environment Variable
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: The GEMINI_API_KEY environment variable is NOT set.")
        print("Please set it using 'export GEMINI_API_KEY=\"YOUR_KEY\"' (Linux/macOS) or 'set GEMINI_API_KEY=\"YOUR_KEY\"' (Windows CMD).")
        return

    # 2. Attempt Authentication and API Call
    try:
        # Client initializes, attempting to use the environment variable
        client = genai.Client()

        # Try a simple, low-cost operation: listing available models
        print("Attempting to list models... (This verifies authentication)")
        models = client.models.list()
        
        # 3. Check the Result
        if any("gemini-2.5-flash" in m.name for m in models):
            print("\n✅ SUCCESS: The API key is active and successfully authenticated.")
            print("The client can communicate with the Gemini API. You are ready to run 04_gemini_solve.py.")
        else:
            print("\n⚠️ WARNING: Authentication was successful, but the list of models was unexpected.")
            print("The key is active, but check your network or usage limits.")
            
    except APIError as e:
        # This error typically means the key is invalid, revoked, or has insufficient permissions/usage quotas.
        print(f"\n❌ ERROR: The API call failed. Key is NOT activated or is invalid.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: An unexpected issue occurred during the check: {e}")

if __name__ == "__main__":
    check_api_key_status()