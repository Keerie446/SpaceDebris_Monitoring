import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

def _load_data(data_source: str) -> pd.DataFrame | None:
    # Handle Data Source Loading
    raw_df = None
    if data_source == "Upload dataset":
        uploaded = st.file_uploader("Upload debris dataset (CSV or Parquet)", type=["csv", "parquet"])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(uploaded)
                else:
                    raw_df = pd.read_parquet(uploaded)
            except Exception as e:
                st.error(f"Failed to read dataset: {e}")
    else:
        # Load Example Dataset
        example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "debris-detection 2"))
        example_files = []
        if os.path.isdir(example_dir):
            for ext in ("csv", "parquet"):
                example_files.extend(glob.glob(os.path.join(example_dir, f"*.{ext}")))

        if example_files:
            example_files = sorted(example_files)
            example_names = [os.path.basename(p) for p in example_files]
            choice = st.selectbox("Choose dataset", options=example_names, label_visibility="collapsed")
            chosen_path = example_files[example_names.index(choice)]
            try:
                if chosen_path.lower().endswith(".csv"):
                    raw_df = pd.read_csv(chosen_path)
                else:
                    raw_df = pd.read_parquet(chosen_path)
            except Exception as e:
                st.error(f"Failed to read example dataset: {e}")
        else:
            st.info("No example datasets found. Switch to 'Upload dataset'.")
    return raw_df

def render_intelligence_view() -> None:
    # Page Header
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; letter-spacing: -1px;">Space Debris Intelligence</h1>
            <p style="font-size: 1.15rem; color: #64748b; margin-top: 0;">Live orbital projection and autonomous risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Source Selection (less prominent)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col_ds1, col_ds2 = st.columns([1, 2])
    with col_ds1:
        data_source = st.radio("Telemetry Source", options=["Example dataset", "Upload dataset"], index=0, horizontal=True)
    with col_ds2:
        raw_df = _load_data(data_source)
    st.markdown("</div>", unsafe_allow_html=True)

    if raw_df is None or raw_df.empty:
        st.stop()

    with st.spinner("Processing Orbital Telemetry..."):
        df = raw_df.copy()

        # Step 1: Clean Dataset
        alt_col = next((c for c in df.columns if 'altitude' in c.lower()), None)
        inc_col = next((c for c in df.columns if 'inclination' in c.lower()), None)
        vel_col = next((c for c in df.columns if 'velocity' in c.lower()), None)
        size_col = next((c for c in df.columns if 'size' in c.lower() or 'area' in c.lower()), None)
        type_col = next((c for c in df.columns if 'type' in c.lower() or 'orbit' in c.lower()), None)
        
        if not alt_col or not inc_col:
            import ast
            objs = []
            if "bboxes" in df.columns:
                for _, row in df.iterrows():
                    bstr = row.get("bboxes")
                    try:
                        boxes = ast.literal_eval(bstr) if isinstance(bstr, str) else []
                    except:
                        boxes = []
                    for b in boxes:
                        area = float(b[2] * b[3]) if len(b) >= 4 else 1.0
                        objs.append({"Object size": max(1.0, area / 100.0)})
            
            if not objs:
                objs = [{"Object size": 1.0} for _ in range(500)]
                
            df = pd.DataFrame(objs)
            rng = np.random.default_rng(42)
            df["Altitude"] = rng.uniform(200, 1200, size=len(df))
            df["Inclination"] = rng.uniform(0, 180, size=len(df))
            
            alt_col = "Altitude"
            inc_col = "Inclination"
            size_col = "Object size"
            st.info("No orbital telemetry found. Synthesized data based on bounding boxes/defaults.")

        if alt_col and inc_col:
            df = df.dropna(subset=[alt_col, inc_col])
            
            # Normalize missing velocity
            if vel_col:
                df[vel_col] = df[vel_col].fillna(df[alt_col].apply(lambda x: 7.8))
            else:
                df['Velocity'] = 7.8
                vel_col = 'Velocity'
                
            if not size_col:
                if 'Object size' not in df.columns:
                    df['Object size'] = 1.0
                size_col = 'Object size'
                
            if not type_col:
                df['Orbit type'] = 'Unknown'
                type_col = 'Orbit type'

            # Step 2: Create Orbital Coordinates (Lat/Lon mapping)
            df["lat"] = df[inc_col] * np.sin(np.linspace(0, 2 * np.pi, len(df)))
            df["lon"] = np.linspace(-180, 180, len(df))

            # Step 3: Risk Score
            df["risk_score"] = (
                0.4 * df[alt_col] +
                0.3 * df[vel_col] +
                0.3 * df[size_col]
            )

            if df["risk_score"].nunique() > 2:
                df["risk_level"] = pd.qcut(df["risk_score"], 3, labels=["Low", "Medium", "High"], duplicates='drop')
            else:
                df["risk_level"] = "Medium"

            # Step 4: Clustering (Hotspot Detection)
            coords = df[["lat", "lon"]]
            n_clusters = min(5, len(df))
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                df["cluster"] = kmeans.fit_predict(coords)
                df["cluster"] = df["cluster"].astype(str)
            else:
                df["cluster"] = "0"

            # 🌍 HERO: Earth Map (full width)
            color_discrete_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
            
            fig = px.scatter_geo(
                df,
                lat="lat",
                lon="lon",
                color="risk_level",
                hover_name=type_col,
                hover_data=[alt_col, vel_col, "risk_score"],
                title="Global Orbital Projection",
                color_discrete_map=color_discrete_map,
                projection="natural earth",
                opacity=0.8,
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    lakecolor="#e0f2fe",
                    showland=True,
                    landcolor="#f1f5f9",
                    showcountries=True,
                    countrycolor="#cbd5e1",
                    projection_scale=1.1
                ),
                font=dict(color="#1e293b"),
                title_font=dict(color="#0f172a", size=24, family="Inter"),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            
            st.markdown("<div class='glass-card' style='padding: 0;'>", unsafe_allow_html=True)
            st.plotly_chart(fig, theme=None, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ⚠️ INSIGHTS SECTION
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>⚠️ Real-Time Insights</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tracked Objects", f"{len(df):,}")
            with col2:
                high_risk = len(df[df["risk_level"] == "High"])
                st.metric("Critical Risk Threats", f"{high_risk:,}")
            with col3:
                avg_alt = df[alt_col].mean()
                st.metric("Avg Altitude (km)", f"{avg_alt:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # 📌 TOP RISK ZONES & 📊 SMALL STATS
            col_zones, col_stats = st.columns(2)
            
            with col_zones:
                st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📌 Top Risk Regions (Clusters)</div>", unsafe_allow_html=True)
                cluster_risk = df.groupby('cluster').agg(
                    Avg_Risk_Score=('risk_score', 'mean'),
                    Threat_Count=('risk_score', 'count')
                ).reset_index()
                cluster_risk = cluster_risk.sort_values(by='Avg_Risk_Score', ascending=False)
                
                # Format for display
                cluster_risk["Region Zone"] = ["Zone Alpha", "Zone Beta", "Zone Gamma", "Zone Delta", "Zone Epsilon"][:len(cluster_risk)]
                cluster_risk["Avg_Risk_Score"] = cluster_risk["Avg_Risk_Score"].round(2)
                
                st.dataframe(
                    cluster_risk[["Region Zone", "Threat_Count", "Avg_Risk_Score"]], 
                    use_container_width=True, 
                    hide_index=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_stats:
                st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📊 Threat Distribution</div>", unsafe_allow_html=True)
                # Small bar chart for risk levels
                risk_counts = df['risk_level'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                fig_bar = px.bar(
                    risk_counts, 
                    x='Risk Level', 
                    y='Count',
                    color='Risk Level',
                    color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                    height=250
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title=None,
                    yaxis_title=None
                )
                st.plotly_chart(fig_bar, theme=None, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("Dataset must contain 'Altitude' and 'Inclination' columns for visualization.")
