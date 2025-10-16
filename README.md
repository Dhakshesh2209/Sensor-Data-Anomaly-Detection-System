# Sensor-Data-Anomaly-Detection-System
Anomaly detection and dashboard using the NASA CMAPSS FD001 dataset
# 🚀 Sensor Data Anomaly Detection System

A predictive maintenance pipeline built using NASA's CMAPSS FD001 dataset. This project detects anomalies in turbofan engine sensor data using the Isolation Forest algorithm, explains model decisions with SHAP, and visualizes results in an interactive Streamlit dashboard.

---

## 🔧 Features

- 📊 Multivariate time-series preprocessing
- 🧠 Rolling statistical feature engineering
- 🔍 Unsupervised anomaly detection (Isolation Forest)
- 📈 SHAP-based model explainability
- 🖥️ Streamlit dashboard for anomaly visualization
- 📉 Remaining Useful Life (RUL) estimation and labeling

---

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib

---

## 📁 Dataset

NASA CMAPSS FD001  
[🔗 Dataset Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 📊 Dashboard Preview

![Sensor Anomaly Detection](images/sensor_anomaly_graph.png)

## 🔍 Model Explainability

![SHAP Waterfall](images/shap_waterfall_chart.png)

---

## 👨‍💻 Author

**C E Dhakshesh**  
B.Tech in Instrumentation Engineering, Class of 2027  
Focused on smart monitoring, predictive control, and software-only instrumentation systems.

---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run dashboard/dashboard_fd001.py

## 📊 Dashboards

### `streamlit_dashboard.py`
Explore internal CMAPSS sequences:
- Sample selector
- RUL prediction
- Anomaly detection
- SHAP sensor importance
- Sensor time-series plots

### `streamlit_csv_dashboard.py`
Upload your own 30×21 `.csv` sensor sequence:
- Predict RUL
- Detect anomalies
- Visualize SHAP impact
- Plot sensor trends

