import ctypes
import numpy as np

def c_array_to_list(c_array, len):
    return [c_array[i] for i in range(len)]

def list_to_c_array(py_list, c_type):
    return (c_type * len(py_list))(*py_list)

def numpy_to_c_3d_array(arr):
    assert arr.ndim == 3, "Input must be a 3D numpy array"
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    
    nrows, ncols, nchannels = arr.shape
    
    ptr_type = ctypes.POINTER(ctypes.c_double)
    data_ptr = arr.ctypes.data_as(ptr_type)
    
    # Create pointers for each row
    row_pointers = (ptr_type * nrows)()
    for i in range(nrows):
        row_pointers[i] = data_ptr + i * ncols * nchannels
    
    # Create pointers for each 2D slice
    slice_pointers = (ctypes.POINTER(ptr_type) * nrows)()
    for i in range(nrows):
        slice_pointers[i] = row_pointers + i
    
    return (ctypes.POINTER(ctypes.POINTER(ptr_type)))(slice_pointers), nrows, ncols, nchannels
