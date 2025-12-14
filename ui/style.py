import streamlit as st


def inject_base_styles() -> None:
    """Apply base layout/typography styles for the Telly app."""

    st.markdown(
        """
        <style>
        :root {
            font-size: 16px;
        }

        html,
        body,
        [data-testid="stAppViewContainer"] {
            font-size: 16px;
        }

        .block-container {
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        [data-testid="stChatInput"] {
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        [data-testid="stChatInput"] > div {
            width: 100%;
            min-height: 5rem;
        }

        [data-testid="stChatInputTextArea"] {
            min-height: 5rem;
        }

        @media (max-width: 1200px) {
            .block-container,
            [data-testid="stChatInput"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_base_styles"]
