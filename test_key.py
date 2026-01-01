import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

print("------------------------------------------------")
print("🔍 CHECKING AVAILABLE MODELS FOR YOUR API KEY...")
print("------------------------------------------------")

try:
    # List all models available to you
    count = 0
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            count += 1
    
    if count == 0:
        print("❌ NO MODELS FOUND! Your API Key might be invalid or restricted.")
    else:
        print(f"\n🎉 Total models found: {count}")

except Exception as e:
    print(f"❌ ERROR CONNECTING: {e}")