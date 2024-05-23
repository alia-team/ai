import ctypes
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

# Define the function signatures of the Rust functions
lib.MLP_new.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.MLP_new.restype = ctypes.POINTER(ctypes.c_void_p)

lib.MLP_propagate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
lib.MLP_propagate.restype = None

lib.MLP_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
lib.MLP_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.MLP_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t, ctypes.c_bool]
lib.MLP_train.restype = None

lib.MLP_free.argtypes = [ctypes.c_void_p]
lib.MLP_free.restype = None

# Create an instance of MLP
npl = (2, 3, 1)  # Exemple de structure de réseau de neurones
npl_arr = (ctypes.c_size_t * len(npl))(*npl)
mlp = lib.MLP_new(npl_arr, len(npl))

# Use the MLP instance
sample_inputs = [1.0, 2.0]
sample_inputs_arr = (ctypes.c_double * len(sample_inputs))(*sample_inputs)
lib.MLP_propagate(mlp, sample_inputs_arr, len(sample_inputs), False)

output_ptr = lib.MLP_predict(mlp, sample_inputs_arr, len(sample_inputs), False)
output = [output_ptr[i] for i in range(npl[-1])]
print(output)

# Free the MLP instance
lib.MLP_free(mlp)
