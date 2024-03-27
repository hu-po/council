import os
import anthropic

anthropic.api_key = os.environ['ANTHROPIC_API_KEY']

# List available models
models = anthropic.list_models()
print([m.slug for m in models])

# Generate using Claude
claude = anthropic.Client("claude-v1")
response = claude.completion(prompt="Hello there", max_tokens_to_sample=1000)
print(response.completion)

# Generate haiku using Haiku
haiku = anthropic.Client("haiku")
response = haiku.completion(prompt="Write a haiku about the ocean", max_tokens_to_sample=50) 
print(response.completion)

# Generate using Opus
opus = anthropic.Client("opus")
response = opus.completion(prompt="What are the key differences between Opus and Claude?", max_tokens_to_sample=500)
print(response.completion)