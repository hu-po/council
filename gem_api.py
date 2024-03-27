import os

import google.generativeai as genai

genai.configure(api_key=os.environ['GOOGLE_SDK_API_KEY'])

print([m.name for m in genai.list_models()])

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello there")
print(response.text)