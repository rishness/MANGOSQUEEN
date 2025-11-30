import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
else:
    print(f"🔑 Key found: {api_key[:5]}... (checking access...)")
    genai.configure(api_key=api_key)

    try:
        print("\n📡 Connecting to Google to list available models...")
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ AVAILABLE: {m.name}")
                found_any = True
        
        if not found_any:
            print("\n⚠️ No 'generateContent' models found. Check if the 'Generative Language API' is enabled in your Google Cloud Console.")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")