"""Use the official OpenAI SDK against the phone. Requires: pip install 'thinai[openai]'."""

from thinai import Thinai

client = Thinai()
oai = client.openai()

model = client.running()[0].name
completion = oai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello from my phone."}],
    max_tokens=40,
)
print(completion.choices[0].message.content)
