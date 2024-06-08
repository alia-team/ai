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

lib = CDLL(f"./target/release/{lib_filename}")

lib.new_rbf.argtypes = [
    POINTER(c_size_t),
    c_size_t,
    c_bool,
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
