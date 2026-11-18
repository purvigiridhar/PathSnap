"""P4: evaluator UI (Streamlit). Run on Laptop 2: API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Shows the ten trips, sends the selected trip_id to the API, displays predicted route + metrics."""
import os
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("PathSnap Evaluator")
st.write(f"API: {API_URL}")
# TODO: fetch /trips, selectbox, POST /predict, render map and results
