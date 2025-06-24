import requests
import json

# API endpoint
url = "https://api.x.ai/v1/chat/completions"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer xai-En0sKQa1BRbZTHzMMTl6kUBfvA26xF1BQvoOk6eFHMTWoUKAXYW3Gz4q7HzYnyHkCQYMXAOW8bHr4Ygt"
}

# Payload
payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "Testing?"
        }
    ],
    "model": "grok-2-1212",
    "stream": False,
    "temperature": 0
}

# Make the request
try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()  # Raise an exception for 4xx/5xx errors
    print("Response Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except requests.RequestException as e:
    print(f"Request failed: {str(e)}")