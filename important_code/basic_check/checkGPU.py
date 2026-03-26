import sys

print(f"Python version: {sys.version}")

# Check PyTorch + CUDA
try:
    import torch
    print(f"\n--- PyTorch ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
        print(f"\nReady to train with PyTorch: YES")
    else:
        print("No CUDA-capable GPU found. Training will use CPU only.")
except ImportError:
    print("PyTorch not installed.")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"\n--- TensorFlow ---")
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU count: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
        print("Ready to train with TensorFlow: YES")
    else:
        print("No GPU found for TensorFlow. Training will use CPU only.")
except ImportError:
    print("TensorFlow not installed.")

# Check nvidia-smi via subprocess
import subprocess
print("\n--- nvidia-smi ---")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("nvidia-smi not available or no NVIDIA GPU detected.")
except FileNotFoundError:
    print("nvidia-smi not found. NVIDIA drivers may not be installed.")
