"""
smoke_test.py — quick LLM connectivity check.

Usage:
    python smoke_test.py

Set LLM_PROVIDER in .env before running.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from graph.llm import get_llm  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "groq")
    print(f"Provider : {provider}")

    llm = get_llm(role="default")
    print(f"Model    : {llm!r}")

    print("Sending test message...")
    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    print(f"Response : {response.content}")
    print("PASS")

if __name__ == "__main__":
    main()
