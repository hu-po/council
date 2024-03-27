import os
import openai

# Configure OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# List available models (optional, just for demonstration)
models = openai.Model.list()
print([model.id for model in models.data])

# Generate content with GPT-4V (assuming 'text-davinci-004' as an example model identifier for GPT-4V)
response = openai.Completion.create(
  model="text-davinci-004", # Replace with the actual model identifier for GPT-4V
  prompt="Hello there",
  max_tokens=50
)

print(response.choices[0].text.strip())
