import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ui.pages.landing import render_landing
from ui.pages.intelligence import render_intelligence_view


def main() -> None:
    st.set_page_config(
        page_title="Space Debris Risk Monitor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Clean UI white theme
    st.markdown(
        """
        <style>
        /* Base styling */
        .stApp {
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6, p, span, div {
            font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
        }
        
        /* Hide sidebar entirely */
        [data-testid="collapsedControl"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2563eb;
            line-height: 1.2;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Cards and containers */
        .glass-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid #f1f5f9;
            margin-bottom: 24px;
        }

        /* Section Headers */
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f1f5f9;
        }

        /* File uploader and selectbox styling */
        [data-testid="stFileUploader"] {
            background-color: white !important;
            color: black !important;
        }
        [data-testid="stSelectbox"] {
            background-color: white !important;
            color: black !important;
        }
        select {
            background-color: white !important;
            color: black !important;
        }
        input[type="file"] {
            background-color: white !important;
            color: black !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Directly render the single intelligence view page
    render_intelligence_view()

if __name__ == "__main__":
    main()

