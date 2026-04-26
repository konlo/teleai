import streamlit as st


def inject_base_styles() -> None:
    """Apply premium design system styles for the Telly app."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;700&display=swap');

        :root {
            --primary: #4f46e5;
            --primary-muted: #6366f1;
            --bg-main: #f9fafb; /* Soft off-white for eye comfort */
            --bg-card: #ffffff;
            --bg-sidebar: #f1f5f9;
            --text-main: #1e293b; /* Professional slate instead of pure black */
            --text-muted: #64748b;
            --border-soft: #e2e8f0;
            color-scheme: light !important;
        }

        /* Global Aesthetics */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="stMain"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-main) !important;
            color: var(--text-main) !important;
            color-scheme: light !important;
        }

        /* Consistent text color */
        .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: var(--text-main) !important;
            line-height: 1.6; /* Improved readability */
        }

        /* Sidebar: Integrated and Soft */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarNav"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-soft) !important;
        }

        /* Hero Section: Professional Gradient */
        .hero-container {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            padding: 2.5rem 2rem;
            border-radius: 1rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .hero-title { color: #ffffff !important; font-weight: 800; font-size: 2.2rem !important; }
        .hero-subtitle { color: rgba(255, 255, 255, 0.9) !important; font-size: 1rem; }

        /* Chat Messages: Clean and Distinct */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 1rem !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }

        /* Chat Input: Floating and Refined */
        .stChatInputContainer {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 1.5rem !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
            padding: 2px 8px !important;
        }

        [data-testid="stChatInputTextArea"] textarea {
            color: var(--text-main) !important;
            background: transparent !important;
        }

        /* Tables & Dataframes */
        [data-testid="stDataFrame"], .stDataFrame, [data-testid="stTable"] {
            border: 1px solid var(--border-soft) !important;
            border-radius: 0.75rem;
            background-color: #ffffff !important;
        }

        /* Popovers / Modals visibility */
        div[data-baseweb="popover"],
        div[data-baseweb="modal"],
        [data-testid="stPopoverBody"],
        [data-testid="stPopoverContent"] {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 0.75rem !important;
        }

        /* Ensure markdown and text inside popover is dark */
        div[data-baseweb="popover"] .stMarkdown,
        div[data-baseweb="popover"] p,
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] label {
            color: #1e293b !important;
        }

        /* Buttons: Muted Indigo / Professional White */
        .stButton button, .stDownloadButton button, [data-testid="stPopover"] button {
            background-color: var(--bg-card) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 0.5rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease;
        }

        .stButton button:hover, .stDownloadButton button:hover, [data-testid="stPopover"] button:hover {
            border-color: var(--primary) !important;
            color: var(--primary) !important;
            background-color: #f5f3ff !important;
        }

        /* Expander / Thinking Log */
        .stExpander {
            border: 1px solid var(--border-soft) !important;
            background-color: #ffffff !important;
            border-radius: 0.75rem !important;
        }

        /* Code Blocks: Soft Background */
        .stCodeBlock, .stCodeBlock div, .stCodeBlock code {
            background-color: #f8fafc !important;
            color: #334155 !important;
            border-radius: 0.5rem !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-main); }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }

        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_base_styles"]
