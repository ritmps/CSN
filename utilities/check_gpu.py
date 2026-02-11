import tensorflow as tf
import os

print('TensorFlow version:', tf.__version__)

print("--- Listing Physical Devices ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Num GPUs Available: {len(gpus)}')
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: Name: {gpu.name}, Device Type: {gpu.device_type}")
else:
    print('Num GPUs Available: 0')
    print('No GPUs detected by TensorFlow (tf.config.list_physical_devices).')

print("--- TensorFlow Built-in CUDA Check ---")
try:
    print(f"TensorFlow is built with CUDA support: {tf.test.is_built_with_cuda()}")
    print(f"CUDA device count (experimental): {tf.config.experimental.list_physical_devices('GPU')}")
except Exception as e:
    print(f"Error during TensorFlow built-in CUDA check: {e}")

print("--- Environment Variables ---")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')}")

print("--- cuDNN Check (via environment) ---")
try:
    from tensorflow.python.platform import build_info
    if hasattr(build_info, 'cuda_version_lib'):
        print(f"TensorFlow found CUDA library version: {build_info.cuda_version_lib}")
    if hasattr(build_info, 'cudnn_version_lib'):
        print(f"TensorFlow found cuDNN library version: {build_info.cudnn_version_lib}")
    else:
        print("TensorFlow's build_info does not report cuDNN version, possibly due to error.")
except ImportError:
    print("Could not import TensorFlow build_info for detailed CUDA/cuDNN version check.")
except Exception as e:
    print(f"Error accessing TensorFlow build_info: {e}")
