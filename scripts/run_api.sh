#!/usr/bin/env bash
# Laptop 1
export EVAL_REFERENCE_DIR="${EVAL_REFERENCE_DIR:-./data/evaluator}"
uvicorn api.main:app --host 0.0.0.0 --port 8000
