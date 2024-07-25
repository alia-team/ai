import ctypes
from ctypes import c_char_p, c_void_p, string_at
import numpy as np
import platform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_processing import get_all_images_in_folder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image


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

    def image_to_vector(image: Image):
        image_array = np.array(image)
        image_vector = image_array.flatten()
        return image_vector
    
    # Example usage:
    images = get_all_images_in_folder("../dataset")
    labels = []
    inputs = []
    for label, image_vector_ptrs in images.items():
        labels += [label] * len(image_vector_ptrs)
        for image_vector_ptr in image_vector_ptrs:
            image_vector = ctypes.cast(image_vector_ptr, ctypes.POINTER(ctypes.c_double))
            inputs.append(np.ctypeslib.as_array(image_vector, (100 * 100 * 1,)))


    npl = (100 * 100 * 1, 3, 3, 3)
    mlp = MLP(npl)
    mlp.init()

    labels = [
        (
            [1.0, -1.0, -1.0]
            if label == "phidippus"
            else [-1.0, 1.0, -1.0] if label == "tegenaria" else [-1.0, -1.0, 1.0]
        )
        for label in labels
    ]

    # Séparer les données en 3 classes
    class_1_inputs = [
        inputs[i] for i in range(len(inputs)) if labels[i] == [1.0, -1.0, -1.0]
    ]
    class_2_inputs = [
        inputs[i] for i in range(len(inputs)) if labels[i] == [-1.0, 1.0, -1.0]
    ]
    class_3_inputs = [
        inputs[i] for i in range(len(inputs)) if labels[i] == [-1.0, -1.0, 1.0]
    ]

    class_1_labels = [
        labels[i] for i in range(len(labels)) if labels[i] == [1.0, -1.0, -1.0]
    ]
    class_2_labels = [
        labels[i] for i in range(len(labels)) if labels[i] == [-1.0, 1.0, -1.0]
    ]
    class_3_labels = [
        labels[i] for i in range(len(labels)) if labels[i] == [-1.0, -1.0, 1.0]
    ]

    # Mélanger les données de chaque classe
    np.random.shuffle(class_1_inputs)
    np.random.shuffle(class_2_inputs)
    np.random.shuffle(class_3_inputs)

    # Diviser chaque classe en ensembles d'entraînement et de test
    train_inputs_1, test_inputs_1, train_labels_1, test_labels_1 = train_test_split(
        class_1_inputs, class_1_labels, test_size=0.2, random_state=42
    )
    train_inputs_2, test_inputs_2, train_labels_2, test_labels_2 = train_test_split(
        class_2_inputs, class_2_labels, test_size=0.2, random_state=42
    )
    train_inputs_3, test_inputs_3, train_labels_3, test_labels_3 = train_test_split(
        class_3_inputs, class_3_labels, test_size=0.2, random_state=42
    )

    # Combiner les ensembles d'entraînement et de test
    training_inputs = train_inputs_1 + train_inputs_2 + train_inputs_3
    training_labels = train_labels_1 + train_labels_2 + train_labels_3
    testing_inputs = test_inputs_1 + test_inputs_2 + test_inputs_3
    testing_labels = test_labels_1 + test_labels_2 + test_labels_3

    # Standardize the dataset
    x_train_mean = np.mean(training_inputs, axis=0)
    x_train_std = np.std(training_inputs, axis=0)

    training_inputs = (training_inputs - x_train_mean) / x_train_std
    testing_inputs = (testing_inputs - x_train_mean) / x_train_std
    print(x_train_std)

    # print("Training...")
    # res = mlp.train(
    #     training_inputs,
    #     training_labels,
    #     testing_inputs,
    #     testing_labels,
    #     0.01,
    #     1000,
    #     True,
    # )
    # print("Saving model...")
    # full_path: str = mlp.save("../models/", "mlp")

    # load the model
    print("Loading model...")
    mlp = load_mlp("../models/mlp_best.json")

    # Predict
    print("Predicting...")
    print(testing_inputs[0])
    prediction = mlp.predict(testing_inputs[0], True)
    print(prediction, "expected:", testing_labels[0])



    # Prédire pour chaque test input
    predictions = np.array([mlp.predict(testing_input, True) for testing_input in testing_inputs])

    cm = confusion_matrix(np.argmax(testing_labels, axis=1), np.argmax(predictions, axis=1))

    # Afficher la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    print("predict image")


    inputs = image_to_vector(Image.open("tmp_img/image.png"))
    image_mean = np.mean(inputs)
    image_std = np.std(inputs)
    print("std", image_std)
    img_input = (inputs - image_mean) / image_std
    print(img_input)
    p = mlp.predict(img_input[0], True)
    print(p)