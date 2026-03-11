from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv


def load_root_env():
    root_dir = next(p for p in Path(__file__).resolve().parents if p.name == "SU-650")
    env_file = str(root_dir / ".env")
    load_dotenv(env_file)


load_root_env()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

# ----------
# Skriv din løsning her
# ----------
