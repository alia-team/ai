import ctypes
import numpy as np
import platform

system = platform.system()
if system == 'Linux':
    lib_filename = 'libai.so'
elif system == 'Darwin':  # macOS
    lib_filename = 'libai.dylib'
elif system == 'Windows':
    lib_filename = 'ai.dll'
else:
    raise RuntimeError(f"Unsupported operating system: {system}")

# Load the Rust shared library
lib = ctypes.CDLL(f"./target/release/{lib_filename}")

# Define argument and return types
lib.create_perceptron.argtypes = [ctypes.c_size_t]
lib.create_perceptron.restype = ctypes.POINTER(ctypes.c_void_p)

lib.fit_perceptron.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]

lib.predict_perceptron.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_perceptron.restype = ctypes.c_double

# Example usage
input_size = 2
perceptron = lib.create_perceptron(input_size)

# Prepare data
data_points = np.array([[0.5, 1.0], [1.5, -1.0]], dtype=np.float64)
class_labels = np.array([1.0, -1.0], dtype=np.float64)

# Convert data points to ctypes arrays
data_points_ctypes = []
for point in data_points:
    data_points_ctypes.append((ctypes.c_double * len(point))(*point))

# Convert to a ctypes array of pointers
data_points_ctypes_ptr = (ctypes.POINTER(ctypes.c_double) * len(data_points_ctypes))(*data_points_ctypes)
class_labels_ctypes = class_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Fit the perceptron
lib.fit_perceptron(perceptron, data_points_ctypes_ptr, class_labels_ctypes, len(data_points), input_size, 100)

# Predict
new_point = np.array([1.0, -0.5], dtype=np.float64)
new_point_ctypes = new_point.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
result = lib.predict_perceptron(perceptron, new_point_ctypes, input_size)
print('Prediction:', result)
