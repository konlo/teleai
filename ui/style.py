import streamlit as st


def inject_base_styles() -> None:
    """Apply premium design system styles for the Telly app."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;700&display=swap');

        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --accent-color: #4f46e5;
            --bg-white: #ffffff;
            --card-bg: rgba(248, 250, 252, 0.8);
            --text-main: #0f172a;
            --text-muted: #64748b;
        }

        /* Global Reset */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-white);
            color: var(--text-main);
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Hero Section */
        .hero-container {
            position: relative;
            padding: 6rem 2rem;
            text-align: center;
            background: var(--primary-gradient);
            border-radius: 1.5rem;
            margin-bottom: 3rem;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .hero-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('https://www.transparenttextures.com/patterns/carbon-fibre.png');
            opacity: 0.1;
        }

        .hero-content {
            position: relative;
            z-index: 10;
        }

        .hero-title {
            font-size: 3.5rem !important;
            margin-bottom: 1rem;
            color: white;
            text-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 2rem;
        }

        /* Feature Grid */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 4rem;
        }

        .feature-card {
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 1rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
            border-color: var(--accent-color);
        }

        .feature-card h3 {
            margin-bottom: 0.75rem;
            color: var(--accent-color);
        }

        /* Buttons */
        .primary-button {
            display: inline-block;
            background: white;
            color: #764ba2;
            padding: 0.75rem 2rem;
            border-radius: 9999px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
        }

        .primary-button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            color: #667eea;
        }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            border-radius: 1rem;
            margin-bottom: 1rem;
            padding: 1rem;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        [data-testid="chatAvatarIcon-user"] {
            background-color: #667eea !important;
        }

        [data-testid="chatAvatarIcon-assistant"] {
            background-color: var(--accent-color) !important;
        }

        .stChatMessage.ad-hoc-callout {
            border-left: 5px solid var(--accent-color);
            background-color: rgba(79, 70, 229, 0.05);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0;
        }

        /* Accent text */
        .accent-text {
            color: var(--accent-color);
            font-weight: 600;
        }

        /* Hide Streamlit components */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_base_styles"]
