# Test Ollama connection
import ollama
import requests
import json

print("Testing Ollama connection...")

# Test 1: Check if we can list models
try:
    models = ollama.list()
    print("✅ ollama.list() works")
    print(f"Available models: {[m['name'] for m in models['models']]}")
except Exception as e:
    print(f"❌ ollama.list() failed: {e}")

# Test 2: Try direct API call
try:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "qwen3:14b",
            "prompt": "Hello",
            "stream": False
        }
    )
    print(f"✅ Direct API call status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"❌ API error: {response.text}")
except Exception as e:
    print(f"❌ Direct API call failed: {e}")

# Test 3: Try ollama.generate()
try:
    response = ollama.generate(model="qwen3:14b", prompt="Hello")
    print("✅ ollama.generate() works")
    print(f"Response: {response['response'][:100]}...")
except Exception as e:
    print(f"❌ ollama.generate() failed: {e}")

# Test 4: Check ollama package version
try:
    import ollama
    print(f"Ollama Python package version: {ollama.__version__}")
except:
    print("Could not determine ollama package version")