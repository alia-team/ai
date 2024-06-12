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
    ctypes.c_size_t,
    ctypes.c_double
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

    def fit(self, training_data, labels, nb_iter, alpha):
        samples_count = len(training_data)
        data_points_ctypes = (ctypes.POINTER(ctypes.c_double) * samples_count)(*[
            np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in training_data
        ])
        labels_ctypes = np.ctypeslib.as_ctypes(np.array(labels, dtype=np.float64))
        lib.fit_perceptron(self.perceptron, data_points_ctypes, labels_ctypes, samples_count, self.input_size, nb_iter, alpha)

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

    perceptron.fit(training_data, labels, 10000, 0.01)
    
    print('Linear simple:')
    for sample, label in zip(training_data, labels):
        result = perceptron.predict(sample)
        print(f'{sample} -> {result} (expected: {label})')

    # Linear Multiple : OK
    input_size = 3
    perceptron = Perceptron(input_size)

    training_data = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    labels = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0]).flatten()

    perceptron.fit(training_data, labels, 10000, 0.1)

    print('Linear multiple:')
    for sample, label in zip(training_data, labels):
        result = perceptron.predict(sample)
        print(f'{sample} -> {result} (expected: {label})')

    # XOR: KO
    input_size = 2
    perceptron = Perceptron(input_size)

    training_data = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    labels = np.array([1, 1, -1, -1])

    perceptron.fit(training_data, labels, 1000000, 0.1)

    print('XOR:')
    for sample, label in zip(training_data, labels):
        result = perceptron.predict(sample)
        print(f'{sample} -> {result} (expected: {label})')

