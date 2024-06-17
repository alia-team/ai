from ctypes import c_char_p, c_double, c_size_t, c_void_p, CDLL, POINTER
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

lib = CDLL(f"../target/release/{lib_filename}")

lib.new_rbf.argtypes = [
    POINTER(c_size_t),
    c_size_t,
    c_char_p,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t
]
lib.new_rbf.restype = c_void_p

lib.fit_rbf.argtypes = [
    c_void_p,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t,
    c_double,
    c_size_t
]
lib.fit_rbf.restype = None

lib.predict_rbf.argtypes = [
    c_void_p,
    POINTER(c_double),
    c_size_t
]
lib.predict_rbf.restype = POINTER(c_double)

lib.free_rbf.argtypes = [c_void_p]
lib.free_rbf.restype = None


class RBF:
    def __init__(
        self,
        neurons_per_layer: list[int],
        activation: str,
        training_dataset: list[list[float]],
        labels: list[list[float]]
    ) -> None:
        self.neurons_per_layer: list[int] = neurons_per_layer
        npl = (c_size_t * len(neurons_per_layer))(*neurons_per_layer)

        self.training_dataset_nrows: int = len(training_dataset)
        self.training_dataset_ncols: int = len(training_dataset[0])
        TrainingDatasetType = POINTER(c_double) * self.training_dataset_nrows
        self.training_dataset = TrainingDatasetType(
            *[np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64))
                for sample in training_dataset]
        )

        self.labels_nrows: int = len(labels)
        self.labels_ncols: int = len(labels[0])
        LabelsType = POINTER(c_double) * self.labels_nrows
        self.labels = LabelsType(
            *[np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64))
                for label in labels]
        )

        activation_c_str = activation.encode("utf-8")

        self.model = lib.new_rbf(
            npl,
            len(neurons_per_layer),
            activation_c_str,
            self.training_dataset,
            self.training_dataset_nrows,
            self.training_dataset_ncols
        )

    def fit(self, gamma: float, max_iterations: int) -> None:
        lib.fit_rbf(
            self.model,
            self.training_dataset,
            self.training_dataset_nrows,
            self.training_dataset_ncols,
            self.labels,
            self.labels_nrows,
            self.labels_ncols,
            gamma,
            max_iterations
        )

    def predict(self, input: list[float]) -> list[float]:
        ctypes_input = np.ctypeslib.as_ctypes(np.array(input, dtype=np.float64))
        output_ptr = lib.predict_rbf(self.model, ctypes_input, len(input))
        return [output_ptr[i] for i in range(self.neurons_per_layer[2])]

    def __del__(self) -> None:
        lib.free_rbf(self.model)


if __name__ == "__main__":
    neurons_per_layer = [2, 4, 1]
    is_classification: bool = True
    training_dataset = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    labels = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    rbf: RBF = RBF(
        neurons_per_layer,
        is_classification,
        training_dataset,
        labels
    )

    gamma: float = 0.01
    max_iterations: int = 100
    rbf.fit(gamma, max_iterations)

    input = [1.0, 1.0]
    print(rbf.predict(input))
