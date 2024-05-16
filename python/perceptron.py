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

class Perceptron:
    def __init__(self, input_size):
        self.perceptron = lib.create_perceptron(input_size)
        self.input_size = input_size

    def fit(self, training_data, labels, nb_iter):
        samples_count = len(training_data)
        data_points_ctypes = (ctypes.POINTER(ctypes.c_double) * samples_count)(*[
            np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in training_data
        ])
        labels_ctypes = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.fit_perceptron(self.perceptron, data_points_ctypes, labels_ctypes, samples_count, self.input_size, nb_iter)

    def predict(self, sample_input):
        sample_input_ctypes = np.ctypeslib.as_ctypes(np.array(sample_input, dtype=np.float64))
        return lib.predict_perceptron(self.perceptron, sample_input_ctypes, self.input_size)

# Example usage
if __name__ == "__main__":
    input_size = 2
    perceptron = Perceptron(input_size)
    
    # Linear simple : OK
    training_data = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    labels = np.array([
        1,
        -1,
        -1
    ])

    perceptron.fit(training_data, labels, 1000)
    
    new_point = [3.0, 3.0]
    result = perceptron.predict(new_point)
    print('Linear simple:', result)

    # Linear Multiple : OK
    input_size = 3
    perceptron = Perceptron(input_size)

    training_data = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    labels = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0]).flatten()

    perceptron.fit(training_data, labels, 10000)

    new_point = [1.5, 1.5]
    result = perceptron.predict(new_point)
    print('Linear multiple:', result)

    # XOR: KO
    input_size = 2
    perceptron = Perceptron(input_size)

    training_data = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    labels = np.array([1, 1, -1, -1])

    perceptron.fit(training_data, labels, 1000000)

    new_point = [0, 1]
    result = perceptron.predict(new_point)
    print('XOR:', result)

