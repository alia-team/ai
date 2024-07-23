import ctypes
from ctypes import c_char_p, c_void_p, string_at
import numpy as np
import platform
import matplotlib.pyplot as plt


# Determine the correct shared library filename based on the operating system
system = platform.system()
if system == "Linux":
    lib_filename = "libai.so"
elif system == "Darwin":  # macOS
    lib_filename = "libai.dylib"
elif system == "Windows":
    lib_filename = "ai.dll"
else:
    raise RuntimeError(f"Unsupported operating system: {system}")

# Load the Rust shared library
lib = ctypes.CDLL(f"../target/release/{lib_filename}")


class TrainResult(ctypes.Structure):
    _fields_ = [
        ("loss_values_ptr", ctypes.POINTER(ctypes.c_double)),
        ("len", ctypes.c_size_t),
        ("inner_len", ctypes.c_size_t),
    ]


# Define the function signatures of the Rust functions
lib.mlp_new.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.mlp_new.restype = ctypes.POINTER(ctypes.c_void_p)

lib.mlp_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_bool,
]
lib.mlp_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.mlp_train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.c_bool,
]
lib.mlp_train.restype = TrainResult

lib.mlp_free.argtypes = [ctypes.c_void_p]
lib.mlp_free.restype = None

lib.mlp_neurons_per_layer.argtypes = [ctypes.c_void_p]
lib.mlp_neurons_per_layer.restype = ctypes.POINTER(ctypes.c_size_t)

lib.mlp_nlayers.argtypes = [ctypes.c_void_p]
lib.mlp_nlayers.restype = ctypes.c_size_t

lib.mlp_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p];
lib.mlp_save.restype = c_char_p

lib.mlp_load.argtypes = [ctypes.c_char_p]
lib.mlp_load.restype = c_void_p


class MLP:
    def __init__(self, npl):
        self.npl_c = (ctypes.c_size_t * len(npl))(*npl)
        self.npl = npl
        self.output_size = npl[-1]
        self.model = None

    def init(self) -> None:
        self.model = lib.mlp_new(self.npl_c, len(self.npl))

    def train(
        self,
        training_dataset,
        training_labels,
        test_dataset,
        test_labels,
        alpha,
        epochs,
        is_classification,
    ):
        samples_count = len(training_dataset)
        sample_inputs_len = len(training_dataset[0])
        tests_count = len(test_dataset)
        test_inputs_len = len(test_dataset[0])

        TrainingDatasetType = ctypes.POINTER(ctypes.c_double) * samples_count
        training_dataset_ctypes = TrainingDatasetType(
            *[
                np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64))
                for sample in training_dataset
            ]
        )

        trainingLabelsType = ctypes.POINTER(ctypes.c_double) * samples_count
        training_labels_ctypes = trainingLabelsType(
            *[
                np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64))
                for label in training_labels
            ]
        )

        testDatasetType = ctypes.POINTER(ctypes.c_double) * tests_count
        test_dataset_ctypes = testDatasetType(
            *[
                np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64))
                for sample in test_dataset
            ]
        )

        testLabelsType = ctypes.POINTER(ctypes.c_double) * tests_count
        test_labels_ctypes = testLabelsType(
            *[
                np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64))
                for label in test_labels
            ]
        )

        train_result = lib.mlp_train(
            self.model,
            training_dataset_ctypes,
            training_labels_ctypes,
            test_dataset_ctypes,
            test_labels_ctypes,
            samples_count,
            sample_inputs_len,
            tests_count,
            test_inputs_len,
            alpha,
            epochs,
            is_classification,
        )
        length = train_result.len
        inner_length = train_result.inner_len
        loss_values_ptr = train_result.loss_values_ptr

        # Convert the pointer to a numpy array
        flat_loss_values = np.ctypeslib.as_array(loss_values_ptr, shape=(length,))
        # Reshape the flat array to the original nested list shape
        loss_values = flat_loss_values.reshape(-1, inner_length).tolist()
        lib.free_train_result(train_result)
        return loss_values

    def save(self, path: str, model_name: str) -> str:
        path_c_str: bytes = path.encode("utf-8")
        model_name_c_str: bytes = model_name.encode("utf-8")
        full_path_c = lib.mlp_save(self.model, path_c_str, model_name_c_str)
        full_path = string_at(full_path_c).decode("utf-8")
        return full_path

    def predict(self, sample_inputs, is_classification):
        sample_inputs_arr = np.ctypeslib.as_ctypes(
            np.array(sample_inputs, dtype=np.float64)
        )
        output_ptr = lib.mlp_predict(
            self.model, sample_inputs_arr, len(sample_inputs), is_classification
        )
        output = [output_ptr[i] for i in range(self.output_size)]
        return output

    def __del__(self):
        lib.mlp_free(self.model)


def load_mlp(model_path: str) -> MLP:
    model = lib.mlp_load(model_path.encode("utf-8"))
    npl_c = lib.mlp_neurons_per_layer(model)
    nlayers = lib.mlp_nlayers(model)
    npl = [npl_c[i] for i in range(nlayers)]
    mlp: MLP = MLP(npl)
    mlp.model = model
    return mlp

if __name__ == "__main__":
    print("Initializing MLP...")
    npl = (2, 4, 1)
    mlp = MLP(npl)
    mlp.init()
    training_dataset = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    labels = np.array([[1], [1], [-1], [-1]])

    print("Fitting...")
    history = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 1000000, True)
 
    print("Saving model...")
    full_path: str = mlp.save("../models/", "mlp_xor")

    print("Loading model...")
    loaded_model: MLP = load_mlp(full_path)

    print('Predicting... It should predict something close to `1`.')
    output: list[float] = loaded_model.predict(training_dataset[0], True)
    print(f"Predicted: {output}")
