import os
import numpy as np
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Smart Home Energy Monitor - Edge AI Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise-grade CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    
    .badge-on {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-off {
        background-color: rgba(107, 114, 128, 0.2);
        color: #9ca3af;
        border: 1px solid #4b5563;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-alert {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .feature-tag {
        background-color: #1e1b4b;
        color: #a5b4fc;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Telemetry scaling factors (extracted from training pipeline)
SCALER_MEAN = np.array([1207.92521, 7.61260393, 0.519932767, 171.412458], dtype=np.float32)
SCALER_STD = np.array([881.928994, 4.74120312, 0.332363057, 99.6844349], dtype=np.float32)

# --- Embedded Model Weights for Pure NumPy Inference ---
W1 = np.array([
    [-1.9253312e-35, 2.375183e-35, -0.39021757, -1.7526667e-05, -3.1559515e-35, -1.1156724e-34, -4.433056e-35, -5.5823526e-35, 4.9741767e-35, 2.7438653e-35, 8.6004436e-35, -0.0023691114, -6.534837e-35, -1.7310526e-35, 1.0402372e-34, -9.974867e-35],
    [-7.651116e-35, -5.3446606e-35, -0.2502486, -6.0222415e-06, -5.4770724e-35, 3.847696e-35, 4.0139484e-35, 1.1405001e-34, -4.578989e-35, -7.2039747e-35, -9.0868296e-36, 0.0016109401, -9.649663e-35, 5.0043176e-35, -6.860453e-36, -3.9132787e-35],
    [1.0845383e-34, -1.0131337e-34, -0.2557293, -1.9136098e-05, -4.8325847e-35, -9.442386e-35, 8.2996335e-35, 3.8463116e-35, 1.6602032e-35, -5.238731e-35, -6.1362366e-35, -0.004884927, -5.0852253e-35, -3.3239645e-35, -3.2021832e-36, -2.291958e-35],
    [6.8806024e-35, -4.9294063e-35, -0.0014958241, -1.334473e-06, 6.36317e-35, 7.007025e-35, 2.5720152e-35, -3.4375404e-36, 4.0989924e-35, -2.7180242e-35, -1.2996298e-35, 0.0010623545, -3.5278606e-37, -4.4854523e-36, 7.909948e-35, 6.491913e-35]
], dtype=np.float32)

b1 = np.array([-3.6465049e-09, -3.607481e-22, 1.0447452, 0.00012861268, -1.8495432e-07, -2.5134679e-11, -6.791822e-08, -1.0341146e-10, -2.144669e-13, -4.3551007e-21, -1.7100825e-09, 0.2093178, -8.122e-08, -9.059567e-08, -5.7374594e-09, -3.8651557e-08], dtype=np.float32)

W2 = np.array([
    [1.0186273e-34, -8.326532e-36, 4.9605144e-35, -8.7825666e-35, -7.35997e-36, 1.0380809e-35, 1.0143125e-34, -3.7818202e-35],
    [1.5152665e-35, 8.368909e-35, 6.7200837e-35, -7.6239384e-35, -8.132648e-35, 5.7650594e-36, 4.9852004e-35, -6.904648e-35],
    [1.0660248e-34, 0.08615767, -3.774563e-35, 0.20830207, -0.43039688, 0.008968685, 0.18380557, -0.0082695205],
    [-8.224259e-36, 2.8782456e-06, 9.110477e-36, 1.2104023e-05, -2.1289672e-05, 1.2921166e-06, 1.10777155e-05, 2.3695452e-06],
    [-9.0219645e-35, 5.9260642e-36, 1.0483177e-34, 3.3432044e-35, -1.1025715e-34, 9.630433e-35, -1.0966788e-34, -3.739631e-35],
    [-4.7081476e-35, -4.1132103e-35, -9.598769e-35, -8.605399e-36, 1.0153433e-34, 2.8813588e-35, 2.1250717e-35, 6.0547963e-35],
    [7.530614e-35, 6.703111e-35, 4.3076174e-35, -3.1442235e-35, -9.978948e-36, -4.4797605e-35, 4.571594e-35, -4.8170456e-35],
    [-6.6130655e-35, -3.406387e-35, -7.345986e-35, 7.731118e-35, 3.8282292e-35, -8.5591704e-35, -8.4092346e-35, 6.621019e-35],
    [7.3128083e-35, 3.8314822e-35, 2.0720613e-35, 4.5138317e-35, -6.171278e-35, 5.275241e-35, 3.6374852e-35, -1.8050737e-35],
    [4.226402e-35, 8.012257e-35, 5.589935e-35, -6.918414e-35, -2.7919056e-35, 4.7513111e-35, -1.9797822e-35, 2.8196413e-35],
    [3.4578693e-35, -2.8309233e-35, -5.9334443e-35, 6.155081e-35, 5.73452e-35, 5.8851854e-35, -9.759806e-36, 4.85086e-35],
    [-1.0232484e-34, 0.0032484317, -5.0535316e-05, 0.0036643005, -0.009052328, 0.0021599825, 0.0035364502, -0.001560281],
    [-6.659299e-35, 7.872461e-35, 6.754893e-35, -8.830029e-35, 9.139075e-35, -2.5930435e-35, 4.1403603e-35, -2.9132228e-36],
    [-1.0243019e-34, -8.2496745e-35, 9.657913e-35, 6.621668e-35, -2.2833278e-35, -1.1573074e-34, 1.6784733e-35, -4.9954584e-35],
    [8.419411e-36, -1.1426048e-34, 1.0176978e-34, 5.081149e-36, -7.5657445e-35, -3.2244086e-35, -5.1502494e-35, -7.7708074e-35],
    [9.289232e-37, -5.00278e-35, -3.1702644e-35, -5.8379676e-35, -5.378699e-35, 1.0212185e-34, 5.9047216e-35, -1.129891e-34]
], dtype=np.float32)

b2 = np.array([-0.05021938, 0.65355915, -0.08521196, -0.122765064, 0.24720299, 2.613551, -0.1087434, 2.339427], dtype=np.float32)

W3 = np.array([
    [0.17535496, 0.65515006, -0.60308295],
    [-15.495121, 1.179195, -16.33521],
    [0.56096435, 0.4320646, 0.32561454],
    [-43.11399, -58.632504, -29.364677],
    [0.08692742, -47.964935, 44.961037],
    [4.546543, 1.2854147, -0.58379173],
    [-29.347458, -53.699474, -21.501865],
    [7.3104095, 0.76066434, 3.0149424]
], dtype=np.float32)

b3 = np.array([2.1259143, 0.29111177, -0.4910851], dtype=np.float32)

def relu(x):
    return np.maximum(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_numpy_inference(scaled_inputs):
    # Layer 1
    h1 = relu(np.dot(scaled_inputs, W1) + b1)
    # Layer 2
    h2 = relu(np.dot(h1, W2) + b2)
    # Output layer
    out = sigmoid(np.dot(h2, W3) + b3)
    return out

# Header layout
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-header">Smart Home Energy Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Edge AI TinyML Simulator & Slab Billing Forecast Platform</div>', unsafe_allow_html=True)
with col_status:
    st.markdown('<div style="text-align: right; padding-top: 1rem;"><span class="badge-on">● LOCAL WI-FI SYSTEM ONLINE</span></div>', unsafe_allow_html=True)

# Navigation tabs
tab_nilm, tab_billing, tab_specs = st.tabs([
    "📟 Live Edge AI Simulator (NILM)", 
    "📊 Intelligent Bill Forecast Engine", 
    "📂 Hardware Specs & packed structs"
])

# ==========================================
# TAB 1: NILM SIMULATOR
# ==========================================
with tab_nilm:
    st.subheader("Simulate Home Load Telemetry")
    st.write("Adjust the sliders below to mimic the aggregate power coming from the PZEM-004T CT sensor. The quantized INT8 TinyML model running in the background will disaggregate the load and predict active appliances instantly.")
    
    col_sliders, col_ai = st.columns([1, 1])
    
    with col_sliders:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔌 PZEM Sensor Emulation")
        
        # Presets to make it extremely easy to use
        preset = st.selectbox("Quick Load Presets", [
            "Select a preset...",
            "Base Idle Load Only (Fridge ON, others OFF)",
            "Water Motor Running (Fridge ON, Motor ON)",
            "Iron Running (Fridge ON, Iron ON, Motor OFF)",
            "Power Leakage Anomaly State (>10W draw, all AI flags OFF)"
        ])
        
        # Default slider coordinates based on preset
        def_power, def_curr, def_pf, def_volt = 344.0, 2.5, 0.60, 227.0
        
        if preset == "Base Idle Load Only (Fridge ON, others OFF)":
            def_power, def_curr, def_pf, def_volt = 345.0, 2.54, 0.60, 227.0
        elif preset == "Water Motor Running (Fridge ON, Motor ON)":
            def_power, def_curr, def_pf, def_volt = 1350.0, 6.2, 0.90, 226.5
        elif preset == "Iron Running (Fridge ON, Iron ON, Motor OFF)":
            def_power, def_curr, def_pf, def_volt = 2050.0, 9.2, 0.95, 226.0
        elif preset == "Power Leakage Anomaly State (>10W draw, all AI flags OFF)":
            # Idle power draws 200W, but no appliance threshold is met
            def_power, def_curr, def_pf, def_volt = 25.0, 0.18, 0.50, 228.0
            
        power = st.slider("Active Power (Watts)", 0.0, 3500.0, float(def_power), 5.0)
        current = st.slider("Current (Amperes)", 0.0, 15.0, float(def_curr), 0.1)
        pf = st.slider("Power Factor (PF)", 0.0, 1.0, float(def_pf), 0.01)
        voltage = st.slider("Line Voltage (Volts)", 150.0, 270.0, float(def_volt), 0.5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overload/Under-voltage flags
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Safety Guard Limits")
        
        # Voltage thresholds
        v_low = st.number_input("Brownout Lower Voltage Limit (V)", 150, 210, 180)
        v_high = st.number_input("Overvoltage Upper Voltage Limit (V)", 230, 270, 250)
        curr_limit = st.number_input("Overload Current Cut-off Limit (A)", 5.0, 15.0, 10.0, 0.5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_ai:
        # Preprocess features and scale inputs
        features = np.array([power, current, pf, voltage], dtype=np.float32)
        scaled_features = (features - SCALER_MEAN) / SCALER_STD
        
        # Run inference using embedded neural network weights
        pred_fridge, pred_motor, pred_iron = run_numpy_inference(scaled_features)
        
        fridge_on = pred_fridge > 0.5
        motor_on = pred_motor > 0.5
        iron_on = pred_iron > 0.5
        
        # Power Leakage Logic
        # Active power > 10W and no appliance is ON
        is_leakage = (power > 10.0) and (not fridge_on) and (not motor_on) and (not iron_on)
        
        # Safety relays logic
        trip_cause = ""
        is_tripped = False
        
        if voltage < v_low:
            is_tripped = True
            trip_cause = f"BROWNOUT FAULT: Voltage dropped below safe threshold ({voltage:.1f}V < {v_low}V)"
        elif voltage > v_high:
            is_tripped = True
            trip_cause = f"OVERVOLTAGE FAULT: Voltage exceeded safe threshold ({voltage:.1f}V > {v_high}V)"
        elif current > curr_limit:
            is_tripped = True
            trip_cause = f"OVERLOAD FAULT: Current exceeded safety threshold ({current:.1f}A > {curr_limit:.1f}A)"
            
        # UI Outputs
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🟢 System Relay State")
        if is_tripped:
            st.markdown(f'<span class="badge-alert">TRIPPED</span>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #ef4444; font-weight: 600; margin-top: 0.5rem;">{trip_cause}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-on">CONNECTED</span>', unsafe_allow_html=True)
            st.markdown('<div style="color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;">Relay closed. Electrical parameter throughput active.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Edge AI Classifier Outputs (INT8 Quantized)")
        
        def show_appliance_row(name, is_on, confidence):
            badge = f'<span class="badge-on">ON</span>' if is_on else f'<span class="badge-off">OFF</span>'
            bar_color = "#10b981" if is_on else "#4b5563"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div>
                    <span style="font-weight: 600; font-size: 1.05rem;">{name}</span>
                    <br><span style="color: #9ca3af; font-size: 0.8rem;">Confidence: {confidence*100:.1f}%</span>
                </div>
                {badge}
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(confidence))
            
        show_appliance_row("Refrigerator", fridge_on, pred_fridge)
        show_appliance_row("Water Motor", motor_on, pred_motor)
        show_appliance_row("Iron / Heavy AC", iron_on, pred_iron)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Power Leakage Detection")
        if is_leakage:
            st.markdown('<span class="badge-alert">⚠️ POWER LEAKAGE DETECTED</span>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #ef4444; font-size: 0.9rem; margin-top: 0.5rem;">Background consumption is {power:.1f}W, but no appliance classifications were resolved. Verify wall sockets for hidden ground leakage immediately.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-on">NORMAL</span>', unsafe_allow_html=True)
            st.markdown('<div style="color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;">No background power leakage detected. Idle loads are within safe margin.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: BILL FORECASTING
# ==========================================
with tab_billing:
    st.subheader("Intelligent Bill Prediction Engine")
    st.write("Simulate the React Native forecasting calculations. By running an OLS Linear Regression over daily consumption, the mobile app forecasts month-end electricity usage and applies WAPDA slabs and taxes to estimate the bill.")
    
    col_bill_in, col_bill_out = st.columns([1, 1])
    
    with col_bill_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Forecast Variables")
        
        budget = st.number_input("Monthly Budget Limit (PKR)", 1000, 50000, 10000, 500)
        current_day = st.slider("Current Day of Month", 2, 29, 15)
        
        # Generate custom log history coordinate
        st.markdown("#### Simulate Daily Cumulative Consumption (kWh)")
        
        growth_type = st.radio("Daily Consumption Profile", [
            "Linear & Consistent (e.g. 10 units/day)",
            "Escalating / Spiky (Increased usage over time)",
            "Energy Saving Mode (Reduced usage over time)"
        ])
        
        # Build dataset points
        days = list(range(1, current_day + 1))
        consumption = []
        
        if growth_type == "Linear & Consistent (e.g. 10 units/day)":
            for d in days:
                consumption.append(d * 10.0)
        elif growth_type == "Escalating / Spiky (Increased usage over time)":
            curr = 0.0
            for d in days:
                increment = 5.0 + (d * 0.8) # Gets larger each day
                curr += increment
                consumption.append(curr)
        elif growth_type == "Energy Saving Mode (Reduced usage over time)":
            curr = 0.0
            for d in days:
                increment = 15.0 - (d * 0.4) # Gets smaller each day
                curr += max(increment, 2.0)
                consumption.append(curr)
                
        # Interactive table data editor
        df_history = pd.DataFrame({
            'Day': days,
            'Cumulative_kWh': consumption
        })
        
        edited_df = st.data_editor(df_history, num_rows="fixed", disabled=["Day"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_bill_out:
        # Perform Linear Regression: y = mx + c
        hist_days = edited_df['Day'].values
        hist_kwh = edited_df['Cumulative_kWh'].values
        
        n = len(hist_days)
        sum_x = np.sum(hist_days)
        sum_y = np.sum(hist_kwh)
        sum_xy = np.sum(hist_days * hist_kwh)
        sum_xx = np.sum(hist_days * hist_days)
        
        denom = n * sum_xx - sum_x * sum_x
        slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict at Day 30
        prediction_kwh = slope * 30.0 + intercept
        current_kwh = hist_kwh[-1]
        prediction_kwh = max(prediction_kwh, current_kwh)
        
        # Calculate WAPDA Tariff Slabs
        # 1-100: 22.0, 101-200: 28.5, 201-300: 34.0, 301-700: 42.0, 700+: 48.0
        slabs = [
            (100, 22.0),
            (200, 28.5),
            (300, 34.0),
            (700, 42.0),
            (float('inf'), 48.0)
        ]
        
        remaining = prediction_kwh
        energy_cost = 0.0
        prev_limit = 0
        
        for limit, rate in slabs:
            capacity = limit - prev_limit
            if remaining > capacity:
                energy_cost += capacity * rate
                remaining -= capacity
                prev_limit = limit
            else:
                energy_cost += remaining * rate
                break
                
        # Apply national taxes
        tax_gst = energy_cost * 0.17
        tax_duty = energy_cost * 0.015
        fixed_charges = 350.0
        total_bill = energy_cost + tax_gst + tax_duty + fixed_charges
        
        # UI Dashboard card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📅 Predicted Bill Dashboard")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f'<div class="metric-label">Forecasted Month-End Usage</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{prediction_kwh:.1f} kWh</div>', unsafe_allow_html=True)
            st.markdown(f'<span style="color: #9ca3af; font-size: 0.8rem;">Current usage: {current_kwh:.1f} kWh</span>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f'<div class="metric-label">Estimated WAPDA Bill</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">Rs. {total_bill:,.0f}</div>', unsafe_allow_html=True)
            
            # Show budget alerts
            if total_bill > budget:
                st.markdown('<span class="badge-alert">⚠️ BUDGET THRESHOLD EXCEEDED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-on">✅ WITHIN BUDGET MARGIN</span>', unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Render Native Streamlit Line Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Ordinary Least Squares Linear Regression Model Fit")
        
        plot_days = np.array(range(1, 31))
        plot_pred = slope * plot_days + intercept
        plot_pred = np.clip(plot_pred, 0, None)
        
        # Build pandas DataFrame for chart plotting (index=Day)
        chart_df = pd.DataFrame(index=plot_days)
        chart_df['Logged Cumulative Usage (kWh)'] = np.nan
        chart_df.loc[hist_days, 'Logged Cumulative Usage (kWh)'] = hist_kwh
        chart_df['OLS Regression Projection (kWh)'] = plot_pred
        
        # Render beautiful responsive native chart
        st.line_chart(chart_df, height=280)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: SPECIFICATIONS
# ==========================================
with tab_specs:
    st.subheader("System Data Specs & Struct Definitions")
    st.write("Details of the memory-optimized binary logging structure used in the ESP32-S3 C++ firmware.")
    
    col_code1, col_code2 = st.columns(2)
    
    with col_code1:
        st.markdown("#### C++ 40-Byte Packed Struct")
        st.code("""
// Strictly packed to exactly 40 bytes to align with SD Card physical block boundaries
struct __attribute__((__packed__)) EnergyTelemetry {
    uint32_t timestamp;      // Epoch Unix time (4 bytes)
    float voltage;           // Line Voltage in Volts (4 bytes)
    float current;           // Current in Amperes (4 bytes)
    float active_power;      // Active Power in Watts (4 bytes)
    float frequency;         // Line Frequency in Hz (4 bytes)
    float power_factor;      // Power Factor 0.0 to 1.0 (4 bytes)
    float energy_kwh;        // Cumulative Energy consumption (4 bytes)
    float temperature;       // Local ambient temperature (4 bytes)
    float humidity;          // Local ambient humidity (4 bytes)
    uint16_t ups_charge;     // UPS backup battery charge percentage (2 bytes)
    uint16_t status_flags;   // Bitwise system state (Relay status, Leakage, AI alert) (2 bytes)
};
        """, language="cpp")
        
    with col_code2:
        st.markdown("#### C++ Feature Scaling Calibration Arrays")
        st.code(f"""
// Precalculated standardization constants for model input normalization
// Input = (RawValue - Mean) / StdDev

const float SCALER_MEAN[4] = {{
    {SCALER_MEAN[0]:.5f},  // Active Power (W)
    {SCALER_MEAN[1]:.5f},  // Current (A)
    {SCALER_MEAN[2]:.5f},  // Power Factor
    {SCALER_MEAN[3]:.5f}   // Average Voltage (V)
}};

const float SCALER_STD[4] = {{
    {SCALER_STD[0]:.5f},  // Active Power (W)
    {SCALER_STD[1]:.5f},  // Current (A)
    {SCALER_STD[2]:.5f},  // Power Factor
    {SCALER_STD[3]:.5f}   // Average Voltage (V)
}};
        """, language="cpp")
        
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>Smart Home Energy Optimization System - Developed for Sessions 2022-2026. Localized Privacy-centric NILM Monitor.</div>", unsafe_allow_html=True)
