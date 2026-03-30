# graph/llm.py
import os
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr


def get_llm(role: str = "default") -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = (
            "llama-3.3-70b-versatile" if role == "critic" else "llama-3.1-8b-instant"
        )
        return ChatGroq(model=model, temperature=0)

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        raw_key = os.getenv("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(raw_key) if raw_key else None,
            temperature=0,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model="gpt-oss:120b-cloud", temperature=0)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)  # type: ignore[call-arg]

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
