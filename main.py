import streamlit as st
from ui.style import inject_base_styles

st.set_page_config(
    page_title="Telly | Telemetry Intelligence",
    page_icon="✨",
    layout="wide",
)

# Apply global styles
inject_base_styles()

def landing_page():
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <h1 class="hero-title">✨ Telemetry Chatbot Telly</h1>
                <p class="hero-subtitle">Your AI-powered expert for SSD telemetry and data analysis.</p>
                <div class="hero-buttons">
                    <a href="/Telly" target="_self" class="primary-button">Launch Dashboard</a>
                </div>
            </div>
        </div>
        
        <div class="feature-grid">
            <div class="feature-card">
                <h3>🔍 SQL Builder</h3>
                <p>Generate optimized Databricks SQL queries from natural language.</p>
            </div>
            <div class="feature-card">
                <h3>📊 EDA Analyst</h3>
                <p>Perform deep exploratory data analysis and anomaly detection with one click.</p>
            </div>
            <div class="feature-card">
                <h3>⚙️ Live Diagnostics</h3>
                <p>Monitor real-time execution logs and intermediate AI thought processes.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Note about the image: The generated hero image is used via CSS in inject_base_styles or directly here if possible.
    # Since I cannot easily set a background image via CSS in Streamlit without a hosted URL or base64, 
    # I will use st.image for the hero background if needed, but a clean gradient often looks better for landing pages.
    
    st.markdown("---")
    st.markdown("### 🚀 Ready to explore your data?")
    if st.button("Enter Telly Lab", use_container_width=True, type="primary"):
        st.switch_page("pages/Telly.py")

if __name__ == "__main__":
    landing_page()