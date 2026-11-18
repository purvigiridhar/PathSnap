@echo off
REM Laptop 2 - Run Streamlit UI
cd %~dp0..
if not defined API_URL set API_URL=http://localhost:8000
echo Starting PathSnap Streamlit UI (API target: %API_URL%) ...
streamlit run ui\app.py
