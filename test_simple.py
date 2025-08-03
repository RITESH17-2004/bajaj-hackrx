import requests
import json

# Test with a working PDF first
url = "http://localhost:8000/hackrx/run"

# Simple test
data = {
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": [
        "What is this document about?",
        "What type of document is this?"
    ]
}

print("Testing with simple PDF...")
response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))

# Now test with the insurance policy URL (without query parameters first)
print("\n" + "="*50)
print("Testing with insurance policy PDF...")

data2 = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What are the covered benefits?"
    ]
}

response2 = requests.post(url, json=data2)
print("Status Code:", response2.status_code)
print("Response:", json.dumps(response2.json(), indent=2))