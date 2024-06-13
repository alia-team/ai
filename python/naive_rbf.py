from ctypes import c_bool, c_double, c_size_t, c_void_p, CDLL, POINTER
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

lib.new_naive_rbf.argtypes = [
    c_size_t,
    c_size_t,
    c_bool,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t
]
lib.new_naive_rbf.restype = c_void_p

lib.fit_naive_rbf.argtypes = [
    c_void_p,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t,
    POINTER(POINTER(c_double)),
    c_size_t,
    c_size_t,
    c_double
]
lib.fit_naive_rbf.restype = None

lib.predict_naive_rbf.argtypes = [
    c_void_p,
    POINTER(c_double),
    c_size_t
]
lib.predict_naive_rbf.restype = POINTER(c_double)

lib.free_naive_rbf.argtypes = [c_void_p]
lib.free_naive_rbf.restype = None

class NaiveRBF:
    def __init__(
        self,
        input_neurons_count: int,
        output_neurons_count: int,
        is_classification: bool,
        training_dataset: list[list[float]],
        labels: list[list[float]]
    ) -> None:
        self.neurons_per_layer: list[int] = [input_neurons_count, len(training_dataset), output_neurons_count];

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

        self.model = lib.new_naive_rbf(
            input_neurons_count,
            output_neurons_count,
            is_classification,
            self.training_dataset,
            self.training_dataset_nrows,
            self.training_dataset_ncols
        )

    def fit(self, gamma: float) -> None:
        lib.fit_naive_rbf(
            self.model,
            self.training_dataset,
            self.training_dataset_nrows,
            self.training_dataset_ncols,
            self.labels,
            self.labels_nrows,
            self.labels_ncols,
            gamma
        )

    def predict(self, input: list[float]) -> list[float]:
        ctypes_input = np.ctypeslib.as_ctypes(np.array(input, dtype=np.float64))

        ctypes_output = lib.predict_naive_rbf(self.model, ctypes_input, len(input))
        return [ctypes_output[i] for i in range(self.neurons_per_layer[2])]

    def __del__(self) -> None:
        lib.free_naive_rbf(self.model)
