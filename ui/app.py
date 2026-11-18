"""P4 - Streamlit UI (runs on Laptop 2).
Start:  API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py
"""
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PathSnap Evaluator", layout="wide", page_icon="🗺️")

st.title("🗺️ PathSnap — Map-Matching Evaluator")
st.caption(f"Connecting to Backend API at: `{API}`")

# Check API health & get trip list
try:
    trips_resp = requests.get(f"{API}/trips", timeout=10)
    if trips_resp.status_code != 200:
        st.error(f"API Error ({trips_resp.status_code}): {trips_resp.text}")
        st.stop()
    trips_data = trips_resp.json()
    trips = trips_data.get("trips", [])
    partition = trips_data.get("partition", "evaluator_test")
    st.info(f"📁 Active Dataset Partition: **`{partition}`** ({len(trips)} trips)")
except Exception as e:
    st.error(f"❌ Cannot reach API at `{API}`. Please verify Laptop 1 IP and port 8000: {e}")
    st.stop()

if not trips:
    st.warning("No evaluation trips found on the API server.")
    st.stop()

col_select, col_btn = st.columns([3, 1])
with col_select:
    trip = st.selectbox("Select Evaluation Trip", trips)

with col_btn:
    st.write("")  # padding
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Running map-matching pipeline for trip `{trip}` on Laptop 1..."):
        try:
            r = requests.post(f"{API}/predict/{trip}", timeout=600)
        except requests.exceptions.Timeout:
            st.error("Request timed out. The pipeline took longer than 600s.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to communicate with API: {e}")
            st.stop()

    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text}")
        st.stop()

    res = r.json()
    metrics = res.get("metrics", {})

    st.subheader("📊 Evaluation Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        top1 = metrics.get("top1_accuracy")
        st.metric("Top-1 Accuracy", f"{top1 * 100:.1f}%" if top1 is not None else "N/A")
    with m_col2:
        cont = metrics.get("route_continuity")
        st.metric("Route Continuity", f"{cont * 100:.1f}%" if cont is not None else "N/A")
    with m_col3:
        st.metric("Reference Points", metrics.get("n_points", "N/A"))
    with m_col4:
        st.metric("Latency", f"{res.get('latency_s', 0):.2f} s")

    # Render interactive Folium Map
    if "map_html" in res and res["map_html"]:
        st.subheader("🗺️ Trajectory & Map-Matching Visualisation")
        components.html(res["map_html"], height=700, scrolling=True)
