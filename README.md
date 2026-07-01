## ☀️ SUNGUARD: Solar Flare Intelligence Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![Hackathon](https://img.shields.io/badge/Bharatiya_Antariksh_Hackathon-2026-orange.svg)

**SUNGUARD** is a real-time, dual-track machine learning pipeline designed to ingest, analyze, and predict solar flare events using cross-spectral X-ray telemetry. Built to mirror the payload constraints of ISRO's **Aditya-L1** observatory (specifically the SoLEXS and HEL1OS instruments), this dashboard bridges deep space data with actionable, explainable AI.

## 🚀 Key Features

* **Real-Time Data Ingestion:** Synchronous streaming of high-frequency solar telemetry (simulating 1-minute epoch NOAA/GOES API data).
* **Track A (Nowcasting):** Instantaneous flare detection mimicking a **1D Temporal Convolutional Network (TCN)**. It analyzes rate-of-change (derivatives) in the Soft X-ray spectrum for zero-latency B/C/M/X classification.
* **Track B (Forecasting):** A multi-horizon probabilistic warning system mimicking a **Temporal Fusion Transformer (TFT)**. It leverages cross-spectral physics features (Spectral Hardness Ratio & Neupert Integrals) to predict eruption risks up to 2 hours in advance.
* **Explainable AI (XAI) Engine:** A diagnostic reasoning panel that removes the "black box" of AI, explicitly telling operators *why* a specific alert was triggered based on real-time astrophysics variables.
* **Stateful Simulation:** Includes a "Historical Replay" and "Inject X-Class Flare" testing mode for presentation and validation.

## 🧠 Physics-Driven Feature Engineering

Instead of treating telemetry as isolated data points, SUNGUARD couples them into a thermodynamic system:
1. **Spectral Hardness Ratio ($R_{H/S}$):** Tracks non-thermal electron acceleration by dividing Hard X-ray flux by Soft X-ray flux.
2. **The Neupert Integral:** Calculates the rolling time-integral of Hard X-rays to accurately predict the thermal peak of Soft X-rays before they physically erupt.

## 💻 Installation & Quick Start

### Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/sunguard-aditya-l1.git](https://github.com/devnori-lab/sunguard-aditya-l1.git)
   cd sunguard-aditya-l1