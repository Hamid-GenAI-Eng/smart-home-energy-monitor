import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Configuration
ML_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(ML_DIR, "nilm_model_quant.tflite")

def verify_quantized_model():
    print("[1/2] Loading quantized TFLite model and test datasets...")
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found at {TFLITE_MODEL_PATH}. Run export_tflite.py first.")
        
    x_test_scaled = np.load(os.path.join(ML_DIR, "x_test_scaled.npy"))
    y_test = np.load(os.path.join(ML_DIR, "y_test.npy"))
    
    # Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output tensors details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details.get('quantization', (0.0, 0))
    output_scale, output_zero_point = output_details.get('quantization', (0.0, 0))
    
    print(f"  Input Quantization Properties: Scale={input_scale}, ZeroPoint={input_zero_point}")
    print(f"  Output Quantization Properties: Scale={output_scale}, ZeroPoint={output_zero_point}")
    
    # Run prediction over all test elements
    y_preds = []
    print("[2/2] Running inference using TFLite Interpreter in INT8 mode...")
    
    for idx, sample in enumerate(x_test_scaled):
        # 1. Quantize input: scale float32 value to int8 range
        # q = round(f / scale) + zero_point
        # Clip to ensure fits in signed int8 [-128, 127]
        quantized_sample = np.round(sample / input_scale) + input_zero_point
        quantized_sample = np.clip(quantized_sample, -128, 127).astype(np.int8)
        
        # Prepare batch input shape (1, 4)
        input_data = np.expand_dims(quantized_sample, axis=0)
        interpreter.set_tensor(input_details['index'], input_data)
        
        # Run execution block
        interpreter.invoke()
        
        # Fetch output predictions
        output_data = interpreter.get_tensor(output_details['index'])[0]
        
        # 2. Dequantize output predictions to float range [0.0, 1.0]
        # f = (q - zero_point) * scale
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Threshold at 0.5
        preds = (dequantized_output > 0.5).astype(int)
        y_preds.append(preds)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} / {len(x_test_scaled)} samples...")
            
    y_preds = np.array(y_preds)
    
    # Generate report
    appliance_names = ['Refrigerator', 'Water Motor', 'Iron/AC']
    print("\n================== QUANTIZED TFLITE REPORT ==================")
    for idx, name in enumerate(appliance_names):
        print(f"\nAppliance Classifier (INT8 Quantized): {name}")
        print(classification_report(y_test[:, idx], y_preds[:, idx], target_names=['OFF', 'ON']))
    print("=============================================================")
    print("Verification complete.")

if __name__ == "__main__":
    verify_quantized_model()
