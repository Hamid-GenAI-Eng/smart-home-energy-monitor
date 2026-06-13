# Smart Home Energy Optimization System ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TinyML-blueviolet.svg)](#)
[![Hardware](https://img.shields.io/badge/Micro-ESP32--S3-blue.svg)](#)
[![Client](https://img.shields.io/badge/Mobile-React%20Native-61dafb.svg)](#)

An enterprise-grade, privacy-centric IoT system designed to eliminate domestic "bill shock" and detect background power leakage. The system runs **Non-Intrusive Load Monitoring (NILM)** disaggregation entirely on-chip using a quantized INT8 TinyML model, tracks data offline via local binary block storage, and forecasts month-end slab tariffs on the companion client app.

---

## 📟 Project Architecture & Layout

```
├── Dataset (1).csv             # Raw multi-meter household telemetry dataset
├── fyp doc final 1 (1).docx    # Approved final year project documentation
├── .gitignore                  # Standard repository exclusions
├── ml/                         # Machine Learning Pipeline (TinyML)
│   ├── train_nilm.py           # Training neural network and SMOTE class balancing
│   ├── export_tflite.py        # Shuffled INT8 post-training quantization
│   ├── verify_tflite.py        # Quantized model prediction and metrics evaluation
│   ├── app.py                  # Streamlit Interactive Web Simulator Dashboard
│   └── model.h                 # Compiled C++ hex byte array model for ESP32-S3
├── firmware/                   # Microcontroller Source Code (C++)
│   └── smart_monitor/
│       ├── telemetry.h         # Memory-packed 40-byte binary logging Struct
│       └── smart_monitor.ino   # ESP32-S3 FreeRTOS core execution sketch
└── app/                        # React Native Mobile Client Code
    ├── billing.ts              # OLS Linear Regression forecasting & slab tariff logic
    └── test_billing.js         # Jest/Node test suite validating billing calculations
```

---

## 🚀 Key Features

*   **Edge AI NILM Classification**: Identifies individual heavy appliance states (Refrigerator, Water Motor, Iron/AC) from the single main incoming line. No distributed smart plugs required.
*   **Dual-Core OS Task Scheduling**: Implemented in C++ on the **ESP32-S3** using **FreeRTOS**. Core 0 handles high-speed UART sensor streaming and TFLite Micro inference. Core 1 handles local file writes and safety limits watchdogs.
*   **Zero-Loss Safety Guard Relays**: Automatically pulls the physical GPIO relay pin in **<500ms** if overvoltage ($>250\text{V}$), brownout ($<180\text{V}$), or current overload ($>10\text{A}$) surges are detected.
*   **Memory-Aligned Block Logging**: Logs data in a packed **40-byte C++ struct** directly to physical sectors on a MicroSD card. This prevents filesystem fragmentation and maximizes SD card write endurance.
*   **WAPDA Slab Billing Forecast**: Mobile app uses **Ordinary Least Squares (OLS) Linear Regression** over daily logged consumption to project day 30 units and applies Pakistani tiered residential slab rates (Rs. 22 to Rs. 48 per unit) including GST and electricity duties in real-time.

---

## 🛠️ Getting Started

### 1. Training & Exporting the TinyML Model
First, install the python dependencies:
```bash
pip install tensorflow imbalanced-learn pandas numpy scikit-learn
```

Run the model training pipeline:
```bash
# 1. Clean outliers and train floating-point neural network
python ml/train_nilm.py

# 2. Convert and quantize model to C++ array (generates ml/model.h)
python ml/export_tflite.py

# 3. Verify quantized accuracy metrics in INT8 mode
python ml/verify_tflite.py
```

### 2. Running the Interactive Web Simulator
Launch the local Streamlit dashboard to simulate PZEM telemetry inputs and see the quantized AI predictions and billing graphs in action:
```bash
pip install streamlit
streamlit run ml/app.py
```

### 3. Running Billing Engine Unit Tests
Validate the regression forecasting and slab pricing logic:
```bash
node app/test_billing.js
```

---

## 🔌 Hardware Configuration

*   **MCU**: ESP32-S3-WROOM-1 (8MB Flash, 512KB SRAM)
*   **Sensor**: PZEM-004T v3.0 AC Active Power Sensor (UART interface, TX/RX)
*   **Logger**: SPI MicroSD Card Reader Slot (MISO/MOSI/SCK/CS)
*   **Relay**: 5V Active-Low isolated Solid State Relay (GPIO 18 trigger)

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
