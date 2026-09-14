"""Find the phone, list its models, and ask a question.

python examples/quickstart.py                 # auto-discover
THINAI_HOST=192.168.1.36 python examples/quickstart.py
"""

from thinai import Thinai

with Thinai() as client:
    print(f"Connected to {client.base_url}")
    loaded = {m.name for m in client.running()}
    for model in client.models():
        print(f"  {'*' if model.name in loaded else ' '} {model.name}")

    response = client.chat("In one sentence, why run an LLM on a phone?", num_predict=80)
    print("\n" + response.content.strip())
    if response.metrics and response.metrics.tokens_per_second:
        print(f"\n[{response.metrics.tokens_per_second:.1f} tokens/s on the phone]")
