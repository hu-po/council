"""
Anthropic Claude Wrapper
https://console.anthropic.com/docs/api/reference
"""

import os
import anthropic
import logging

log = logging.getLogger(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def list_models() -> list[str]:
    return ["claude-3-opus-20240229", "claude-3-haiku-20240229"]

def text(prompt: str, model: str = "claude-3-opus-20240229", max_tokens: int = 256):
    response = client.messages.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
    )
    response = response.content
    log.info(f"\n---\nAnthropic {model}\n---\nprompt: {prompt}\nresponse: {response}\n---")
    return response

def test():
    log.debug("Testing Anthropic Claude Wrapper")
    for model in list_models():
        log.debug(model)
    text("What is your name")

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    test()