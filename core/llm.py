import os
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def load_llm(
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """Initialise the Gemini chat model using environment configuration."""
    load_dotenv()
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


__all__ = ["load_llm"]

