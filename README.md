# PathSnap: Distributed Map-Matching & Route Reconstruction

**UE24CS352A Machine Learning Hackathon**  
Pipeline: Noisy GPS $\rightarrow$ Candidate road segments $\rightarrow$ Classifier scores $\rightarrow$ HMM/Viterbi $\rightarrow$ Reconstructed route.  
Served via a FastAPI backend (Laptop 1) and an interactive Streamlit UI (Laptop 2).

---

## 1. Installation

```bash
# 1. Clone repository
git clone https://github.com/purvigiridhar/PathSnap.git
cd PathSnap

# 2. Set up virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 2. Running EDA

The EDA inventory analyzes trips across `train` and `student_test` partitions (never evaluator data).

```bash
# Run script directly:
python -c "from pathsnap.config import load_config; print('Config loaded')"

# Run interactive Jupyter notebook:
# Open notebooks/01_eda.ipynb in VS Code or JupyterLab
```

Precomputed EDA artifacts are stored in:
- `results/eda/dataset_inventory.csv` (Per-trip statistics for 90 trips)
- `results/eda/summary_stats.csv` (Partition aggregate statistics)
- `reports/figures/eda_distributions.png` (GPS points, arc counts, time gap distributions)
- `reports/maps/train_00000046.html` (Interactive Folium map with Esri tiles)

---

## 3. P1 Candidate & Feature Dataset Build

The candidate builder extracts the eight closest road segments per observation using a trip-centred azimuthal equidistant projection.

```powershell
python -m pathsnap.build_dataset data/raw/train data/processed/offline/sample --partition --limit 5 --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/train data/processed/offline/train --partition --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/student_test data/processed/offline/student_test --partition --labeled --local-crs --max-candidates 8
python -m pathsnap.build_dataset data/raw/student_test data/processed/prediction/student_test --partition --local-crs --max-candidates 8
```

For live API inference:
```python
from pathsnap.build_dataset import build_candidates_for_trip
from pathsnap.data.loader import load_track
from pathsnap.data.network import load_projected_network

candidates = build_candidates_for_trip(trip_dir)
track = load_track(trip_dir)
nodes, arcs, crs = load_projected_network(trip_dir)
```

---

## 4. Running the API (Laptop 1)

Runs the FastAPI backend and serves `/health`, `/trips`, and `/predict/{trip_id}`.

### Windows:
```powershell
.venv\Scripts\Activate.ps1
$env:EVAL_DATA_DIR = "data/student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Or double-click: scripts\run_api.bat
```

### Linux / macOS:
```bash
export EVAL_DATA_DIR="data/student_test"
bash scripts/run_api.sh
```

---

## 5. Running the UI (Laptop 2)

Connects to the Laptop 1 API and provides an interactive map-matching dashboard.

### Windows:
```powershell
.venv\Scripts\Activate.ps1
$env:API_URL = "http://<LAPTOP1_IP>:8000"
streamlit run ui/app.py
# Or run: scripts\run_ui.bat
```

### Linux / macOS:
```bash
export API_URL="http://<LAPTOP1_IP>:8000"
bash scripts/run_ui.sh
```

---

## 6. Two-Laptop Deployment

1. Connect both Laptop 1 and Laptop 2 to the same Wi-Fi network or mobile hotspot.
2. Find Laptop 1's local IP address:
   - **Windows:** Run `ipconfig` (look for IPv4 Address, e.g., `192.168.1.45`).
   - **Linux / macOS:** Run `ip a` or `ifconfig`.
3. Allow port 8000 through the firewall on Laptop 1.
4. Verify connectivity from Laptop 2 by opening `http://<LAPTOP1_IP>:8000/health` in a browser (returns `{"status":"ok"}`).
5. Launch Streamlit UI on Laptop 2 pointing to Laptop 1 IP.

---

## 7. Repository Layout

| Path | Purpose |
|---|---|
| `api/` | FastAPI backend, model inference, and server-side evaluation store |
| `ui/` | Streamlit web client for live demo and evaluation |
| `notebooks/` | EDA (`01_eda.ipynb`) and experiment notebooks |
| `src/pathsnap/viz/` | Folium interactive maps (`folium_maps.py`) |
| `reports/` | Output distribution figures (`reports/figures/`) and Folium HTML maps (`reports/maps/`) |
| `results/` | Output tables from EDA and pipeline runs |
| `docs/` | `data_quality_log.md`, `ai_usage_disclosure.md`, `interfaces.md` |
| `scripts/` | One-click launch scripts for API and UI (.sh and .bat) |
| `tests/` | Contract and integration tests |

---

## 8. AI Assistance Disclosure

Full disclosure of AI tool usage is recorded in [`docs/ai_usage_disclosure.md`](docs/ai_usage_disclosure.md).
