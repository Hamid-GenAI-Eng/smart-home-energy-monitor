import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Configuration
ML_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(ML_DIR)
CSV_PATH = os.path.join(WORKSPACE_DIR, "Dataset (1).csv")
MODEL_SAVE_PATH = os.path.join(ML_DIR, "nilm_model.keras")
SCALER_PATH = os.path.join(ML_DIR, "scaler_params.txt")

def load_and_preprocess_data():
    print("[1/5] Loading and cleaning dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}. Please verify file location.")
        
    df = pd.read_csv(CSV_PATH)
    
    # 1. Clean outliers (drop raw power spikes > 5kW, likely sensor errors)
    initial_len = len(df)
    df = df[
        (df['meter1_power'] < 5000) & 
        (df['meter2_power'] < 5000) & 
        (df['meter3_power'] < 5000)
    ]
    print(f"  Cleaned {initial_len - len(df)} noise outliers.")
    
    # 2. Engineer aggregate input features (what a single-phase sensor sees)
    print("  Engineering aggregate features...")
    df['agg_power'] = df['meter1_power'] + df['meter2_power'] + df['meter3_power']
    df['agg_current'] = df['meter1_current'] + df['meter2_current'] + df['meter3_current']
    
    # Weighted average power factor, handling division by zero
    total_pf_power = df['meter1_power'] * df['meter1_pf'] + \
                     df['meter2_power'] * df['meter2_pf'] + \
                     df['meter3_power'] * df['meter3_pf']
    df['agg_pf'] = np.where(df['agg_power'] > 0, total_pf_power / df['agg_power'], 0.0)
    df['agg_pf'] = np.clip(df['agg_pf'], 0.0, 1.0)
    
    # Average line voltage
    df['avg_voltage'] = (df['meter1_voltage'] + df['meter2_voltage'] + df['meter3_voltage']) / 3.0
    
    # 3. Create Multi-Label Targets
    # Refrigerator is represented by Meter 1 being active (base power is around 344W)
    # Water Motor is represented by Meter 3 being in its medium-high power state (~1000W)
    # Iron/AC is represented by Meter 3 being in its peak power state (~1700W)
    print("  Creating appliance targets...")
    df['target_fridge'] = (df['meter1_power'] > 100).astype(int)
    df['target_motor'] = ((df['meter3_power'] > 800) & (df['meter3_power'] < 1200)).astype(int)
    df['target_iron'] = (df['meter3_power'] > 1500).astype(int)
    
    print(f"  Class Distribution:")
    print(f"    Refrigerator ON: {df['target_fridge'].sum()} / {len(df)} ({df['target_fridge'].mean()*100:.2f}%)")
    print(f"    Water Motor ON:  {df['target_motor'].sum()} / {len(df)} ({df['target_motor'].mean()*100:.2f}%)")
    print(f"    Iron/AC ON:      {df['target_iron'].sum()} / {len(df)} ({df['target_iron'].mean()*100:.2f}%)")
    
    features = ['agg_power', 'agg_current', 'agg_pf', 'avg_voltage']
    targets = ['target_fridge', 'target_motor', 'target_iron']
    
    return df[features], df[targets]

def handle_imbalance_and_split(X, y):
    print("[2/5] Splitting data and handling class imbalances via SMOTE...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construct joint decimal states (0 to 7) to synthesize overlap combinations using standard SMOTE
    joint_state = y_train['target_fridge'] * 4 + y_train['target_motor'] * 2 + y_train['target_iron']
    counts = joint_state.value_counts()
    
    # Filter classes with at least 6 samples to satisfy SMOTE k_neighbors=3 requirement
    valid_classes = counts[counts >= 6].index
    mask = joint_state.isin(valid_classes)
    
    X_train_filtered = X_train[mask]
    joint_state_filtered = joint_state[mask]
    
    smote = SMOTE(k_neighbors=3, random_state=42)
    X_train_res, joint_res = smote.fit_resample(X_train_filtered, joint_state_filtered)
    
    # Reconstruct multi-label targets from resampled joint states
    y_train_res = pd.DataFrame()
    y_train_res['target_fridge'] = (joint_res >= 4).astype(int)
    y_train_res['target_motor'] = (((joint_res % 4) >= 2)).astype(int)
    y_train_res['target_iron'] = ((joint_res % 2) == 1).astype(int)
    
    print(f"  Resampled Training set size: {len(X_train_res)} (original: {len(X_train)})")
    
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler parameters for C++ firmware usage
    print(f"  Saving feature scaling parameters for C++ array headers...")
    with open(SCALER_PATH, "w") as sf:
        sf.write("// Feature Scaling Constants for C++ firmware input preprocessing\n")
        sf.write(f"const float SCALER_MEAN[4] = {{{', '.join([str(m) for m in scaler.mean_])}}};\n")
        sf.write(f"const float SCALER_STD[4] = {{{', '.join([str(s) for s in scaler.scale_])}}};\n")
    print(f"  Scaler parameters successfully written to: {SCALER_PATH}")
    
    # Save validation datasets for quantization calibration and verification
    np.save(os.path.join(ML_DIR, "x_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(ML_DIR, "x_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(ML_DIR, "y_test.npy"), y_test.values)
    
    return X_train_scaled, X_test_scaled, y_train_res.values, y_test.values

def build_model(input_dim, output_dim):
    print("[3/5] Compiling Keras feedforward MLP architecture...")
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(output_dim, activation='sigmoid') # Sigmoid activation for independent multi-labels
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    print("[4/5] Training neural network model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=50, # Set to 50 to optimize execution duration
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("[5/5] Evaluating performance on test partition...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    appliance_names = ['Refrigerator', 'Water Motor', 'Iron/AC']
    print("\n================== CLASSIFICATION REPORT ==================")
    for idx, name in enumerate(appliance_names):
        print(f"\nAppliance Classifier: {name}")
        print(classification_report(y_test[:, idx], y_pred[:, idx], target_names=['OFF', 'ON']))
    print("===========================================================")
        
    model.save(MODEL_SAVE_PATH)
    print(f"Floating-point model saved to: {MODEL_SAVE_PATH}")

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = handle_imbalance_and_split(X, y)
    model = build_model(X_train.shape[1], y_train.shape[1] if len(y_train.shape) > 1 else 1)
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
