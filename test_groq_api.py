# test_groq_api.py
# Quick test to check if Groq API key works

from groq import Groq
from dotenv import load_dotenv
import os
import sys

print("=== Groq API Test Script ===")

# Step 1: Load .env file
load_dotenv()

# Step 2: Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found!")
    print(" - Make sure you have a file named '.env' (not .env.example)")
    print(" - Inside it should be: GROQ_API_KEY=your_key_here")
    print(" - No spaces, no quotes around the key")
    sys.exit(1)

print(f"API Key loaded successfully (starts with: {api_key[:6]}...)")

# Step 3: Initialize Groq client
try:
    client = Groq(api_key=api_key)
    print("Groq client initialized OK")
except Exception as e:
    print("ERROR: Failed to create Groq client")
    print(str(e))
    sys.exit(1)

# Step 4: Make a tiny test API call
print("\nSending test message to Llama 3.1 (8B instant)...")

try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say exactly: 'Groq is working perfectly!'",
            }
        ],
        model="llama-3.1-8b-instant",   # small & fast model for testing
        temperature=0.0,
        max_tokens=20,
    )

    response = chat_completion.choices[0].message.content.strip()
    print("\nSuccess! Groq response:")
    print("→", response)

    if "Groq is working perfectly!" in response:
        print("\nTEST PASSED ✓ API key and connection are working!")
    else:
        print("\nResponse was received but content is unexpected.")

except Exception as e:
    print("\nERROR: Groq API call failed!")
    print("Full error:")
    print(str(e))
    if "401" in str(e):
        print("→ Most likely: Invalid or revoked API key")
    elif "429" in str(e):
        print("→ Rate limit hit — wait a few minutes")
    elif "connect" in str(e).lower() or "timeout" in str(e).lower():
        print("→ Network issue — check internet / firewall / VPN")
    sys.exit(1)

print("\nDone. You can now run: python app.py")