import ctypes
import numpy as np
import platform
import matplotlib.pyplot as plt

system = platform.system()
if system == 'Linux':
    lib_filename = 'libai.so'
elif system == 'Darwin':  # macOS
    lib_filename = 'libai.dylib'
elif system == 'Windows':
    lib_filename = 'ai.dll'
else:
    raise RuntimeError(f"Unsupported operating system: {system}")

lib = ctypes.CDLL(f"./target/release/{lib_filename}")

lib.new_linear_regression.argtypes = [ctypes.c_size_t]
lib.new_linear_regression.restype = ctypes.c_void_p

lib.fit_linear_regression.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.fit_linear_regression.restype = None

lib.predict_linear_regression.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_linear_regression.restype = ctypes.c_double

lib.free_linear_regression.argtypes = [ctypes.c_void_p]
lib.free_linear_regression.restype = None

class LinearRegression:
    def __init__(self, input_size):
        self.lr = lib.new_linear_regression(input_size)
        self.input_size = input_size

    def fit(self, training_data, labels):
        samples_count = len(training_data)
        data_points_ctypes = (ctypes.POINTER(ctypes.c_double) * samples_count)(*[
            np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in training_data
        ])
        labels_ctypes = np.ctypeslib.as_ctypes(np.array(labels, dtype=np.float64))
        lib.fit_linear_regression(self.lr, data_points_ctypes, labels_ctypes, samples_count, self.input_size)

    def predict(self, sample_input):
        sample_input_ctypes = np.ctypeslib.as_ctypes(np.array(sample_input, dtype=np.float64))
        return lib.predict_linear_regression(self.lr, sample_input_ctypes, self.input_size)
    
    def __del__(self):
        lib.free_linear_regression(self.lr)

# Example usage
if __name__ == "__main__":
    input_size = 1
    lr = LinearRegression(input_size)
    
    training_data = np.array([[i] for i in range(101)])
    labels = np.array([i * 9 / 5.0 + 32 + np.random.random() * 50 - 25 for i in range(101)])
    lr.fit(training_data, labels)
    responses = [lr.predict([i]) for i in range(101)]
    # Plot the training data and the regression line
    plt.scatter(training_data, labels)
    plt.plot(range(101), responses, color='red')
    plt.show()

    del lr
