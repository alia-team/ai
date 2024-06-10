import ctypes
import numpy as np
import platform
from data_processing import get_all_images_in_folder

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

class TrainResult(ctypes.Structure):
    _fields_ = [("loss_values_ptr", ctypes.POINTER(ctypes.c_double)),
                ("len", ctypes.c_size_t),
                ("inner_len", ctypes.c_size_t)]
# Define the function signatures of the Rust functions
lib.mlp_new.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.mlp_new.restype = ctypes.POINTER(ctypes.c_void_p)

lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
lib.mlp_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.mlp_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t, ctypes.c_bool]
lib.mlp_train.restype = TrainResult

lib.mlp_free.argtypes = [ctypes.c_void_p]
lib.mlp_free.restype = None


class MLP:
    def __init__(self, npl):
        npl_arr = (ctypes.c_size_t * len(npl))(*npl)
        self.mlp = lib.mlp_new(npl_arr, len(npl))
        self.output_size = npl[-1]

    def train(self, training_dataset, training_labels, test_dataset, test_labels, alpha, nb_iter, is_classification):
        samples_count = len(training_dataset)
        sample_inputs_len = len(training_dataset[0])
        tests_count = len(test_dataset)
        test_inputs_len = len(test_dataset[0])
        
        TrainingDatasetType = ctypes.POINTER(ctypes.c_double) * samples_count
        training_dataset_ctypes = TrainingDatasetType(*[np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in training_dataset])
        
        trainingLabelsType = ctypes.POINTER(ctypes.c_double) * samples_count
        training_labels_ctypes = trainingLabelsType(*[np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64)) for label in training_labels])
        
        testDatasetType = ctypes.POINTER(ctypes.c_double) * tests_count
        test_dataset_ctypes = testDatasetType(*[np.ctypeslib.as_ctypes(np.array(sample, dtype=np.float64)) for sample in test_dataset])
        
        testLabelsType = ctypes.POINTER(ctypes.c_double) * tests_count
        test_labels_ctypes = testLabelsType(*[np.ctypeslib.as_ctypes(np.array(label, dtype=np.float64)) for label in test_labels])
        
        train_result  = lib.mlp_train(self.mlp, training_dataset_ctypes, training_labels_ctypes,test_dataset_ctypes, test_labels_ctypes, samples_count, sample_inputs_len, tests_count, test_inputs_len, alpha, nb_iter, is_classification)
        length = train_result.len
        inner_length = train_result.inner_len
        loss_values_ptr = train_result.loss_values_ptr


        # Convert the pointer to a numpy array
        flat_loss_values = np.ctypeslib.as_array(loss_values_ptr, shape=(length,))
        # Reshape the flat array to the original nested list shape
        loss_values = flat_loss_values.reshape(-1, inner_length).tolist()
        lib.free_train_result(train_result)
        return loss_values


    def predict(self, sample_inputs, is_classification):
        sample_inputs_arr = np.ctypeslib.as_ctypes(np.array(sample_inputs, dtype=np.float64))
        output_ptr = lib.mlp_predict(self.mlp, sample_inputs_arr, len(sample_inputs), is_classification)
        output = [output_ptr[i] for i in range(self.output_size)]
        return output
    
    def __del__(self):
        lib.mlp_free(self.mlp)

# # Example usage:
# images = get_all_images_in_folder("datatest")
# labels = []
# inputs = []
# for label, image_vector_ptrs in images.items():
#     labels += [label] * len(image_vector_ptrs)
#     for image_vector_ptr in image_vector_ptrs:
#         image_vector = ctypes.cast(image_vector_ptr, ctypes.POINTER(ctypes.c_double))
#         inputs.append(np.ctypeslib.as_array(image_vector, (100 * 100 * 3,)))

# if __name__ == "__main__":
#     npl = (100 * 100 * 3, 2, 3)
#     mlp = MLP(npl)

#     labels = [[1.0, -1.0, -1.0] if label == 'phidippus' else [-1.0, 1.0, -1.0] if label == 'tegenaria' else [-1.0, -1.0, 1.0] for label in labels]
    
#     print("Training...")
#     mlp.train(inputs, labels, 0.1, 10000, True)
    
#     print("Predicting...")
#     for k in range(len(inputs)):
#         output = mlp.predict(inputs[k], True)
#         print("k:", k, output, labels[k])
