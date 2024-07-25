from ctypes import string_at, c_char_p, c_double, c_longlong, c_size_t, c_void_p, CDLL, POINTER
import numpy as np
import platform
from util import c_array_to_list, list_to_c_array

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


lib.new_cnn.argtypes = [
    c_size_t,   # batch_size
    c_size_t,   # epochs
    c_char_p,   # optimizer
    c_double,   # learning_rate
    c_double,   # optimizer_param2
    c_double    # optimizer_param3
]
lib.new_cnn.restype = c_void_p

lib.set_input_shape.argtypes = [
    c_void_p, # cnn_ptr
    POINTER(c_size_t) # input_shape
]
lib.set_input_shape.restype = None

lib.add_conv2d_layer.argtypes = [
    c_void_p, # cnn_ptr
    c_size_t, # nfilters
    c_size_t  # kernel_size
]
lib.set_input_shape.restype = None

lib.add_maxpool2d_layer.argtypes = [
    c_void_p, # cnn_ptr
    c_size_t  # kernel_size
]
lib.add_maxpool2d_layer.restype = None

lib.add_dense_layer.argtypes = [
    c_void_p, # cnn_ptr
    c_size_t, # output_size
    c_char_p, # activation
    c_double, # dropout
    c_char_p  # weights_init
]
lib.add_dense_layer.restype = None

lib.fit_cnn.argtypes = [
    c_void_p, # cnn_ptr
    c_char_p,   # dataset_path
    c_double,   # train_ratio
    c_longlong, # image_per_class
]
lib.fit_cnn.restype = None

lib.predict_cnn.argtypes = [
    c_void_p, # cnn_ptr
    c_char_p # image_path
]
lib.predict_cnn.restype = POINTER(c_double)

lib.save_cnn.argtypes = [
    c_void_p, # cnn_ptr
    c_char_p, # path
    c_char_p  # model_name
]
lib.save_cnn.restype = c_char_p

lib.load_cnn.argtypes = [
    c_char_p # model_path
]
lib.load_cnn.restype = c_void_p

lib.free_cnn.argtypes = [
    c_void_p # cnn_ptr
]
lib.free_cnn.restype = None

class CNN:
    def __init__(self) -> None:
        self.output_size: int = 0
        self.model = None

    def setup(
        self,
        batch_size: int,
        epochs: int,
        optimizer: str,
        learning_rate: float,
        optimizer_param2: float,
        optimizer_param3: float
    ) -> None:
        optimizer_c_str: bytes = optimizer.encode("utf-8")
        self.model = lib.new_cnn(
            batch_size,
            epochs,
            optimizer_c_str,
            learning_rate,
            optimizer_param2,
            optimizer_param3 
        )

    def set_input_shape(self, input_shape: list[int]):
        input_shape_c = list_to_c_array(input_shape, c_size_t)
        lib.set_input_shape(self.model, input_shape_c)

    def add_conv2d_layer(self, nfilters: int, kernel_size: int):
        lib.add_conv2d_layer(self.model, nfilters, kernel_size)

    def add_maxpool2d_layer(self, kernel_size: int):
        lib.add_maxpool2d_layer(self.model, kernel_size)

    def add_dense_layer(self, output_size: int, activation: str, dropout: float, weights_init: str):
        self.output_size = output_size
        activation_c_str: bytes = activation.encode("utf-8")
        weights_init_c_str: bytes = weights_init.encode("utf-8")
        lib.add_dense_layer(self.model, output_size, activation_c_str, dropout, weights_init_c_str)

    def fit(
        self,
        dataset_path: str,
        train_ratio: float,
        image_per_class: int = -1,
    ) -> None:
        dataset_path_c_str: bytes = dataset_path.encode("utf-8")
        lib.fit_cnn(
            self.model,
            dataset_path_c_str,
            train_ratio,
            image_per_class
        )

    def predict(self, image_path: str) -> list[float]:
        output = lib.predict_cnn(
            self.model,
            image_path.encode("utf-8")
        )
        return c_array_to_list(output, self.output_size)

    def save(self, path: str, model_name: str) -> str:
        path_c_str: bytes = path.encode("utf-8")
        model_name_c_str: bytes = model_name.encode("utf-8")
        full_path_c = lib.save_cnn(self.model, path_c_str, model_name_c_str)
        full_path = string_at(full_path_c).decode("utf-8")
        return full_path

    def free(self) -> None:
        lib.free_cnn(self.model)


def load_cnn(model_path: str) -> CNN:
    cnn = CNN()
    cnn.model = lib.load_cnn(model_path.encode("utf-8"))
    cnn.output_size = 3 # To refactor so we get it dynamically
    return cnn

if __name__ == "__main__":
    print("Initializing CNN...")
    cnn = CNN()
    batch_size: int = 30
    epochs: int = 10
    optimizer: str = "adam"
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    cnn.setup(
        batch_size,
        epochs,
        optimizer,
        learning_rate,
        beta1,
        beta2
    )
    cnn.set_input_shape([100, 100, 1]);
    cnn.add_conv2d_layer(8, 3);
    cnn.add_maxpool2d_layer(2);
    cnn.add_dense_layer(128, "relu", 0.1, "he");
    cnn.add_dense_layer(64, "relu", 0.1, "he");
    cnn.add_dense_layer(3, "softmax", 0., "xavier");

    print("Fitting...")
    dataset_path: str = "../dataset/"
    train_ratio: float = 0.8
    cnn.fit(
        dataset_path,
        train_ratio,
    )

    print("Saving model...")
    full_path: str = cnn.save("../models/", "cnn_b2_999_btch_30")
    print("Freeing CNN...")
    cnn.free()
    print("Freed.")

    print("Loading model...")
    loaded_cnn: CNN = load_cnn(full_path)

    print('Predicting... It should predict "phidippus".')
    image_path: str = "../dataset/phidippus/835255150-388.png"
    output: list[float] = loaded_cnn.predict(image_path)
    predicted: str = ""
    match output.index(max(output)):
        case 0:
            predicted = "avicularia"
        case 1:
            predicted = "phidippus"
        case 2:
            predicted = "tegenaria"
        case _:
            predicted = "error during prediction"
    print(f"Predicted: {predicted}")

    print("Freeing loaded CNN...")
    loaded_cnn.free()
    print("Freed.")
