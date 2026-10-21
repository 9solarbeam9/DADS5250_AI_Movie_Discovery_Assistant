import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(override=True)
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
client = OpenAI()


def parse[T: BaseModel](response_model: type[T], system: str, user: str) -> T:
    reply = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=response_model,
    )
    return reply.choices[0].message.parsed
