import os
import numpy as np
import tensorflow as tf

# Configuration
ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(ML_DIR, "nilm_model.keras")
TFLITE_OUTPUT_PATH = os.path.join(ML_DIR, "nilm_model_quant.tflite")
HEADER_OUTPUT_PATH = os.path.join(ML_DIR, "model.h")

def quantize_and_export():
    print("[1/3] Loading Keras floating-point model...")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Keras model not found at {MODEL_LOAD_PATH}. Please run train_nilm.py first.")
        
    model = tf.keras.models.load_model(MODEL_LOAD_PATH)
    
    # Load calibration data (X_train scaled samples)
    x_train_scaled = np.load(os.path.join(ML_DIR, "x_train_scaled.npy"))
    
    print("[2/3] Executing INT8 Post-Training Quantization...")
    # Setup TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set model input/output to INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Shuffled diverse calibration selection to cover all state changes
    def representative_dataset_gen():
        # Select 300 random indices from the training set to cover all states
        indices = np.random.choice(len(x_train_scaled), size=300, replace=False)
        for idx in indices:
            sample = x_train_scaled[idx]
            yield [np.array([sample], dtype=np.float32)]
            
    converter.representative_dataset = representative_dataset_gen
    
    # Convert model
    tflite_quant = converter.convert()
    
    # Save the TFLite binary file
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_quant)
    print(f"  Quantized TFLite binary saved to: {TFLITE_OUTPUT_PATH} ({len(tflite_quant) / 1024:.2f} KB)")
    
    print("[3/3] Generating C++ byte array header (model.h)...")
    # Convert binary to hex byte array formatted for C++ inclusion
    hex_lines = []
    for i, byte in enumerate(tflite_quant):
        if i % 12 == 0:
            hex_lines.append("\n  ")
        hex_lines.append(f"0x{byte:02x}, ")
        
    header_content = f"""// Quantum-optimized TinyML model byte array for TensorFlow Lite Micro
// Generated automatically from Keras trained neural network model.

#ifndef NILM_MODEL_H
#define NILM_MODEL_H

// Quantized model representation
const unsigned char g_nilm_model_data[] = {{{"".join(hex_lines)}
}};

// Model details
const int g_nilm_model_data_len = {len(tflite_quant)};

#endif // NILM_MODEL_H
"""
    with open(HEADER_OUTPUT_PATH, "w") as h_file:
        h_file.write(header_content)
        
    print(f"  C++ header exported successfully to: {HEADER_OUTPUT_PATH}")
    print(f"  Byte array length: {len(tflite_quant)} bytes ({len(tflite_quant)/1024:.2f} KB)")

if __name__ == "__main__":
    quantize_and_export()
