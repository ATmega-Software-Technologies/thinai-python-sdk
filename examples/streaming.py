"""A tiny terminal chat that streams replies from the phone, sync and async."""

import asyncio

from thinai import AsyncThinai, Thinai


def sync_chat() -> None:
    client = Thinai()
    history = [{"role": "system", "content": "You are a helpful, concise assistant."}]
    print(f"Chatting with {client.base_url} (empty line to quit)")
    while True:
        prompt = input("\nyou> ").strip()
        if not prompt:
            break
        history.append({"role": "user", "content": prompt})
        reply = ""
        print("ai> ", end="")
        for chunk in client.chat(history, stream=True):
            print(chunk.content, end="", flush=True)
            reply += chunk.content
        print()
        history.append({"role": "assistant", "content": reply})


async def async_once() -> None:
    async with AsyncThinai() as client:
        async for chunk in await client.chat("Give me three Tamil greetings.", stream=True):
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(async_once())
    sync_chat()
