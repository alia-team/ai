use crate::activation::{str_to_activation, Activation};
use crate::data::{load_image_dataset, Dataset3D};
use crate::model::{Hyperparameters, CNN};
use crate::optimizer::{str_to_optmizer, Optimizer};
use crate::util::c_str_to_rust_str;
use crate::weights_init::{str_to_weights_init, WeightsInit};
use core::slice;
use libc::{c_char, c_double, c_longlong, size_t};
use ndarray::Array1;
use std::ffi::CString;

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `dataset_path` and `optimizer` are valid, non-null pointers to null-terminated C strings
/// - The returned pointer is freed using `free_cnn` to avoid memory leaks
#[no_mangle]
pub unsafe extern "C" fn new_cnn(
    dataset_path: *const c_char,
    train_ratio: c_double,
    image_per_class: c_longlong,
    batch_size: size_t,
    epochs: size_t,
    optimizer: *const c_char,
    learning_rate: c_double,
    optimizer_param2: c_double,
    optimizer_param3: c_double,
) -> *mut CNN {
    // Load dataset
    let path: &str = c_str_to_rust_str(dataset_path);
    let image_per_class: Option<usize> = match image_per_class < 0 {
        true => None,
        false => Some(image_per_class as usize),
    };
    let dataset: Dataset3D = load_image_dataset(path, train_ratio, image_per_class)
        .expect("Failed to load image dataset.");

    // Setup optimizer
    let optimizer_str: &str = c_str_to_rust_str(optimizer);
    let optimizer: Optimizer = str_to_optmizer(
        optimizer_str,
        learning_rate,
        optimizer_param2,
        optimizer_param3,
    );

    // Build hyperparameters struct
    let hyperparameters: Hyperparameters = Hyperparameters {
        batch_size,
        epochs,
        optimizer,
    };

    Box::leak(Box::new(CNN::new(dataset, hyperparameters)))
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
/// - `input_shape` is a valid pointer to an array of 3 `size_t` elements
#[no_mangle]
pub unsafe extern "C" fn set_input_shape(cnn_ptr: *mut CNN, input_shape: *const size_t) {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    let input_shape: Vec<usize> = unsafe { slice::from_raw_parts(input_shape, 3).to_vec() };
    cnn.set_input_shape(input_shape);
}

/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
#[no_mangle]
pub unsafe extern "C" fn add_conv2d_layer(
    cnn_ptr: *mut CNN,
    nfilters: size_t,
    kernel_size: size_t,
) {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    cnn.add_conv2d_layer(nfilters, kernel_size);
}

/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
#[no_mangle]
pub unsafe extern "C" fn add_maxpool2d_layer(cnn_ptr: *mut CNN, kernel_size: size_t) {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    cnn.add_maxpool2d_layer(kernel_size);
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
/// - `activation` and `weights_init` are valid, non-null pointers to null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn add_dense_layer(
    cnn_ptr: *mut CNN,
    output_size: size_t,
    activation: *const c_char,
    dropout: c_double,
    weights_init: *const c_char,
) {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    let dropout: Option<f64> = if dropout == 0. { None } else { Some(dropout) };
    let activation: Box<dyn Activation> = str_to_activation(c_str_to_rust_str(activation));
    let weights_init: WeightsInit = str_to_weights_init(c_str_to_rust_str(weights_init));
    cnn.add_dense_layer(output_size, activation, dropout, weights_init);
}

/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
#[no_mangle]
pub unsafe extern "C" fn fit_cnn(cnn_ptr: *mut CNN) {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    cnn.fit();
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid, non-null pointer to a CNN struct
/// - `image_path` is a valid, non-null pointer to a null-terminated C string
/// - The returned pointer must not be freed directly; it will be freed when the CNN is freed
#[no_mangle]
pub unsafe extern "C" fn predict_cnn(
    cnn_ptr: *mut CNN,
    image_path: *const c_char,
) -> *const c_double {
    let cnn: &mut CNN = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    let image_path: &str = c_str_to_rust_str(image_path);
    let output: Array1<f64> = cnn.predict(image_path);
    let output_ptr: *const f64 = output.as_ptr();
    std::mem::forget(output);

    output_ptr
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `cnn_ptr`, `path`, and `model_name` are valid, non-null pointers to null-terminated C strings
/// - The returned pointer must be freed by the caller using an appropriate deallocation function
#[no_mangle]
pub unsafe extern "C" fn save_cnn(
    cnn_ptr: *mut CNN,
    path: *const c_char,
    model_name: *const c_char,
) -> *const c_char {
    let cnn = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    let path: &str = c_str_to_rust_str(path);
    let model_name: &str = c_str_to_rust_str(model_name);
    let full_path: CString =
        CString::new(cnn.save(path, model_name)).expect("Failed to convert Rust str to C str.");
    full_path.into_raw()
}

/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - `model_path` is a valid, non-null pointer to a null-terminated C string
/// - The returned pointer must be freed using `free_cnn` to avoid memory leaks
#[no_mangle]
pub unsafe extern "C" fn load_cnn(model_path: *const c_char) -> *mut CNN {
    Box::leak(Box::new(CNN::load(c_str_to_rust_str(model_path))))
}

/// # Safety
///
/// This function is unsafe because it deallocates a raw pointer.
/// The caller must ensure that:
/// - `cnn_ptr` is a valid pointer returned by `new_cnn` or `load_cnn`
/// - `cnn_ptr` has not been freed before
/// - `cnn_ptr` is not used after this function call
#[no_mangle]
pub unsafe extern "C" fn free_cnn(cnn_ptr: *mut CNN) {
    let _ = unsafe { Box::from_raw(cnn_ptr) };
}
