import streamlit as st


def render_landing() -> None:
    st.markdown(
        """
        <div class="neon-title">
            <h1>Space Debris Risk Monitor</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        """
        Upload a debris catalog (CSV/Parquet), visualize orbital density, and predict collision risk zones.
        """
    )

    st.markdown(
        """
        <div class="neon-card">
            <b>What you get</b>
            <ul>
                <li>Debris density heatmaps (altitude vs inclination)</li>
                <li>Clustering visualization (orbital crowding patterns)</li>
                <li>Risk prediction (Low / Medium / High)</li>
                <li>Orbital region risk mapping (3D / animated view)</li>
                <li>Dashboard analytics (counts and trends)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Go to the Dashboard tab from the sidebar to upload your dataset and generate visualizations."
    )

