import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel


def load_llm(
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Initialise the chat model using environment configuration."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "azure":
        return _load_azure_llm(temperature=temperature, max_tokens=max_tokens)
    return _load_google_llm(model=model, temperature=temperature, max_tokens=max_tokens)


def _load_google_llm(
    *, model: str, temperature: float, max_tokens: Optional[int]
) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2,
        disable_streaming=False,
    )


def _load_azure_llm(*, temperature: float, max_tokens: Optional[int]) -> AzureChatOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    missing_fields = [
        name
        for name, value in {
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_DEPLOYMENT_NAME": deployment,
        }.items()
        if not value
    ]
    if missing_fields:
        st.error(
            "Azure OpenAI 연결을 위해 다음 환경 변수를 설정하세요: "
            + ", ".join(missing_fields)
        )
        st.stop()

    kwargs = {
        "azure_endpoint": endpoint,
        "azure_deployment": deployment,
        "api_version": api_version,
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": 2,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return AzureChatOpenAI(**kwargs)


__all__ = ["load_llm"]

