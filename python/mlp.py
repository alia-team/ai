import ctypes
import numpy as np
import platform

# Determine the correct shared library filename based on the operating system
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
lib.mlp_new.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.mlp_new.restype = ctypes.POINTER(ctypes.c_void_p)

lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
lib.mlp_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.mlp_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t, ctypes.c_bool]
lib.mlp_train.restype = None

lib.mlp_free.argtypes = [ctypes.c_void_p]
lib.mlp_free.restype = None

class MLP:
    def __init__(self, npl):
        npl_arr = (ctypes.c_size_t * len(npl))(*npl)
        self.mlp = lib.mlp_new(npl_arr, len(npl))
        self.output_size = npl[-1]

    def train(self, training_dataset, labels, alpha, nb_iter, is_classification):
        samples_count = len(training_dataset)
        sample_inputs_len = len(training_dataset[0])
        
        TrainingDatasetType = ctypes.POINTER(ctypes.c_double) * samples_count
        training_dataset_ctypes = TrainingDatasetType(*[np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in training_dataset])
        
        LabelsType = ctypes.POINTER(ctypes.c_double) * samples_count
        labels_ctypes = LabelsType(*[np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64)) for label in labels])
        
        lib.mlp_train(self.mlp, training_dataset_ctypes, labels_ctypes, samples_count, sample_inputs_len, alpha, nb_iter, is_classification)
    
    def predict(self, sample_inputs, is_classification):
        sample_inputs_arr = np.ctypeslib.as_ctypes(np.array(sample_inputs, dtype=np.float64))
        output_ptr = lib.mlp_predict(self.mlp, sample_inputs_arr, len(sample_inputs), is_classification)
        output = [output_ptr[i] for i in range(self.output_size)]
        return output
    
    def __del__(self):
        lib.mlp_free(self.mlp)

# Example usage:
if __name__ == "__main__":
    npl = (2, 1)
    mlp = MLP(npl)
    
    training_dataset = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ]
    labels = [
        [-1.0],
        [1.0],
        [1.0]
    ]
    
    mlp.train(training_dataset, labels, 0.1, 1000000, True)
    
    for k in range(len(training_dataset)):
        output = mlp.predict(training_dataset[k], True)
        print(output)
