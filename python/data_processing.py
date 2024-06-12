import ctypes
import platform
import os

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
lib = ctypes.CDLL(f"./target/release/{lib_filename}")

# Define the function signatures of the Rust functions
lib.image_to_vector.argtypes = [ctypes.c_char_p]
lib.image_to_vector.restype = ctypes.POINTER(ctypes.c_double)


def image_to_vector(image_path):
    ### Converts an image to a vector
    ### Returns a pointer to the vector
    c_image_path = ctypes.c_char_p(os.fsencode(image_path))
    return lib.image_to_vector(c_image_path)


def get_all_images_in_folder(folder_path):
    ### Get all images in subfolders
    ### Returns a dictionary with the image vectors and their corresponding labels
    images = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_vector = image_to_vector(image_path)
                label = os.path.basename(root)
                if label not in images:
                    images[label] = []
                images[label].append(image_vector)
    return images


def convert_rgb_vectors_to_palette(image_vectors):
    ### Converts RGB vectors to palette vectors
    palette_vectors = []
    for image_vector in image_vectors:
        palette_vector = []
        for i in range(0, len(image_vector), 3):
            r, g, b = image_vector[i], image_vector[i + 1], image_vector[i + 2]
            color = int((r * 6 / 256) * 36 + (g * 6 / 256) * 6 + (b * 6 / 256))
            palette_vector.append(color)
        palette_vectors.append(palette_vector)
    return palette_vectors