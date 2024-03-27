"""
OpenAI GPT Wrapper
https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
"""

import os
import openai
from openai import OpenAI
import logging

log = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def list_models() -> list[str]:
  return [m.id for m in openai.models.list()]

def text(prompt: str, model: str = "gpt-4-1106-preview"):
  response = client.chat.completions.create(
      messages=[{"role": "user", "content": prompt}],
      model=model,
  )
  response = response.choices[0].message.content
  log.info(f"\n---\nOpenAI {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
  return response

def test():
  log.debug("Testing OpenAI GPT Wrapper")
  for model in list_models():
      log.debug(model)
  text("What is your name?")

if __name__ == "__main__":
  log.setLevel(logging.DEBUG)
  handler = logging.StreamHandler()
  handler.setLevel(logging.DEBUG)
  log.addHandler(handler)
  test()