# 🚦 Smart Traffic Signal Control Using a Neuro-Fuzzy System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.4+-4B8BBE?style=for-the-badge)](https://github.com/scikit-fuzzy/scikit-fuzzy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> An adaptive, intelligent traffic signal management system bridging human-interpretable **Mamdani Fuzzy Logic** and high-precision **Deep Neural Networks** to dynamically optimize green light duration, prioritize emergency vehicles, and eliminate urban congestion.

---

## 🌟 Key Features

- **🧠 Mamdani Fuzzy Inference System (FIS):** 28 linguistic expert rules with triangular and trapezoidal membership functions and Centroid defuzzification.
- **⚡ Deep Multi-Output MLP Neural Network:** 5-layer architecture ($4 \to 64 \to 128 \to 64 \to 32 \to 3$) trained on 5,000 diurnal traffic scenarios with Batch Normalization, Dropout, and Early Stopping ($R^2 = 0.9534$).
- **🔀 Adaptive Weighted Ensemble:** Dynamic fusion blending safety-critical deterministic fuzzy rules ($\alpha = 0.70$) for emergency preemption with empirical neural network optimization ($\alpha = 0.35$) during standard flow.
- **🚨 Emergency Vehicle Preemption:** Immediate signal override allocating priority $5.0$ and maximal clearance window to approaching emergency services.
- **🏙️ Interactive Simulation Dashboard:** Live browser-based 4-way intersection simulation featuring dynamic vehicle spawning, countdown timers, SVG circular gauges, active fuzzy rule highlights, and a 12-plot analytical gallery.

---

## 📊 System Benchmark

Benchmarked on 5,000 real-world simulated traffic scenarios:

| Target Variable | System Architecture | MSE | RMSE | MAE | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **🟢 Green Duration (s)** | Mamdani Fuzzy | 76.17 | 8.73 s | 7.02 s | 0.8007 |
| | Neural Network (MLP) | **17.47** | **4.18 s** | **3.25 s** | **0.9543** |
| | **Hybrid Neuro-Fuzzy** | **31.36** | **5.60 s** | **4.29 s** | **0.9179** |
| **⚡ Signal Priority (1–5)** | Mamdani Fuzzy | 0.57 | 0.75 | 0.60 | 0.4796 |
| | Neural Network (MLP) | **0.04** | **0.21** | **0.14** | **0.9595** |
| | **Hybrid Neuro-Fuzzy** | **0.20** | **0.45** | **0.30** | **0.8143** |
| **🔴 Congestion Level (%)** | Mamdani Fuzzy | 136.20 | 11.67 % | 8.10 % | 0.6839 |
| | Neural Network (MLP) | **20.81** | **4.56 %** | **3.63 %** | **0.9517** |
| | **Hybrid Neuro-Fuzzy** | **62.39** | **7.90 %** | **5.40 %** | **0.8552** |
| **🎯 OVERALL SYSTEM** | **Neural Network** | **12.88** | **3.59** | **2.38** | **0.9534** |

---

## 🏗️ Architecture

```
[Sensory Inputs] ──► 1. Vehicle Count (0–100)
                 ──► 2. Traffic Density (0–100%)
                 ──► 3. Waiting Time (0–180s)
                 ──► 4. Emergency Vehicle (0 or 1)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Mamdani Fuzzy Logic   │      │  Deep Neural Network   │
│  • 28 Expert Rules     │      │  • 4→64→128→64→32→3    │
│  • Centroid Defuzz.    │      │  • ReLU + BatchNorm    │
└───────────┬────────────┘      └───────────┬────────────┘
            │ (Y_fuzzy)                     │ (Y_nn)
            └───────────────┬───────────────┘
                            ▼
              ┌───────────────────────────┐
              │  Adaptive Fusion Layer    │
              │  Y = α·Fuzzy + (1-α)·NN   │
              │  α ∈ [0.35, 0.70]         │
              └─────────────┬─────────────┘
                            ▼
[Control Outputs] ◄── 1. Green Light Duration (10–120s)
                  ◄── 2. Signal Priority (1–5 Scale)
                  ◄── 3. Congestion Level (0–100%)
```

---

## 📁 Repository Structure

```
.
├── dashboard/                      # Web simulation & visual interface
│   ├── index.html                  # Live 4-way intersection UI
│   ├── style.css                   # Glassmorphic dark styling
│   ├── script.js                   # Client-side fuzzy inference & physics
│   └── plots/                      # Real-time copies of 12 analytical plots
├── data/
│   └── traffic_data.csv            # 5,000 synthetic traffic samples
├── models/
│   ├── traffic_nn_model.keras      # Trained multi-output Keras model
│   ├── scaler_X.pkl                # Feature MinMax scaler
│   ├── scaler_y.pkl                # Target MinMax scaler
│   └── metrics.json                # Saved evaluation metrics
├── outputs/
│   ├── metrics/                    # System benchmark summaries
│   │   ├── system_comparison.csv
│   │   └── full_summary.json
│   └── plots/                      # 12 publication-ready analytical plots
├── src/
│   ├── data_generator.py           # Traffic distribution generator
│   ├── fuzzy_system.py             # Mamdani FIS implementation
│   ├── neural_network.py           # Multi-output Keras MLP
│   ├── neuro_fuzzy.py              # Adaptive weighted ensemble
│   └── visualizations.py           # Plotting engine
├── index.html                      # Root redirect for GitHub Pages deployment
├── main.py                         # End-to-end execution pipeline
├── PROJECT_REPORT.md               # Detailed academic project report
├── VIVA_VOCE_GUIDE.md              # 25+ viva voce Q&A preparation guide
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ishan565/<REPO_NAME>.git
cd <REPO_NAME>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes data generation, fuzzy initialization, neural network training (150 epochs with early stopping), comparative evaluation, and exports all 12 plots:
```bash
python3 main.py
```

### 3. Launch the Interactive Dashboard Locally
```bash
python3 -m http.server 8085 --directory dashboard
```
Open [http://localhost:8085](http://localhost:8085) in your web browser.

---

## 🌐 Deploying to GitHub Pages

1. In your GitHub repository, navigate to **Settings** > **Pages**.
2. Under **Build and deployment** > **Source**, select `Deploy from a branch`.
3. Select branch: `main` and folder: `/ (root)`.
4. Click **Save**. Within 1–2 minutes, your live simulation will be accessible at:
   `https://ishan565.github.io/<REPO_NAME>/`

---

## 📚 Documentation & Viva Voce
- [Full Academic Project Report](PROJECT_REPORT.md)
- [Viva Voce Examination & Oral Defense Guide](VIVA_VOCE_GUIDE.md)

---

## 📄 License
This project is licensed under the MIT License.
