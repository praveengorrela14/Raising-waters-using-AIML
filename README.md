# Raising-waters-using-AIML
# HydroGuard AP // ML Flood Risk Command Center

HydroGuard AP is an intelligent, high-utility flood prediction and early-warning disaster response dashboard tailored for **Andhra Pradesh, India**. Powered by an enterprise-grade supervised machine learning pipeline (XGBoost core yielding **96.55% accuracy**), the platform ingests meteorological metrics and translates them into localized GIS telemetry mappings and actionable incident logs.

---

## ⚡ Key Core Workspaces

- **Risk Profiler Matrix**: Ingests automated raw sensory metrics (Annual Rainfall, Cloud Visibility, and Seasonal Patterns).
- **GIS Telemetry Mapping Workspace**: Powered by Leaflet.js to automatically lock, refocus, and track real-time visual anomalies across high-risk sectors (Godavari Basin, Krishna Delta, Coastal Vizag Belt).
- **Indicator Vector Engine**: Real-time analytical bar graphs plotting raw metadata variations dynamically.
- **Active Incident Registry Logs**: A persistent local database tracker keeping transactional session records of past risk thresholds.

---

## 🏗️ Project Architecture Layout

Ensure your project space matches this file path distribution:

```text
RASING WATERS/
│
├── app.py                 # Optimized Flask backend server script
├── train_model.py         # Supervised ML algorithm training pipeline
├── requirements.txt       # Unified Python dependencies container
├── Dockerfile             # Containerized architecture specification
├── README.md              # Project documentation deployment guide
└── templates/
    └── index.html         # Fluid CSS Grid/Flex command interface
```

---

## 🚀 Step-by-Step Local Deployment

### 1. Environment Initialization
Navigate to your project directory inside your terminal command center and isolate your environment variables:

```bash
# Move into folder space
cd "C:\Users\admin\Desktop\RASING WATERS"

# Generate a virtual environment
python -m venv venv

# Activate the workspace environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Dependency Management
Install the necessary requirements layer:

```bash
pip install -r requirements.txt
```

### 3. Trigger Machine Learning Pipeline
Generate your scalable binaries (`best_flood_model.pkl` and `scaler.pkl`) by evaluating Decision Trees, Random Forests, KNN, and XGBoost:

```bash
python train_model.py
```

### 4. Boot up Telemetry Console
Launch the local development engine:

```bash
python app.py
```
Open your secure dashboard interface via browser portals:
- Primary Host: `http://127.0.0.1:5000`
- Network Relay: `http://10.86.67.133:5000`

---

## 🛰️ Monsoonal Scenario Validation Tracking

### Extreme Anomaly Workflow:
1. Assign **Target Sector** to `Godavari Basin (Rajahmundry)`.
2. Input **Annual Rainfall** as `8000`.
3. Input **Cloud Visibility** as `56`.
4. Input **Seasonal Rainfall Score** as `89`.
5. Execute prediction pipeline.
6. **Result Metric**: The interface issues a 🔴 `CRITICAL FLOOD WARNING` banner, maps spatial focus markers natively to Andhra Pradesh coordinates `[17.0005, 81.8040]`, and appends the tracking status down to the active incident grid matrix rows.

---

## ☁️ IBM Cloud Deployment Integration

This configuration comes built-in with dynamic port mappings (`os.environ.get('PORT', 8080)`) for container deployment via the **IBM Cloud Code Engine**:

```bash
# Log into IBM Cloud terminal space
ibmcloud login

# Target resource group
ibmcloud target -g Default

# Build and deploy the container image registry directly
ibmcloud ce app create --name hydroguard-ap --image docker.io/YOUR_DOCKERHUB_ID/hydroguard-ap --port 8080
```
