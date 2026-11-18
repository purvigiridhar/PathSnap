@echo off
REM Laptop 1 - Run FastAPI backend
cd %~dp0..
if not defined EVAL_DATA_DIR set EVAL_DATA_DIR=data\student_test
if not defined OPENBLAS_NUM_THREADS set OPENBLAS_NUM_THREADS=1
echo Starting PathSnap API Server on http://0.0.0.0:8000 ...
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
