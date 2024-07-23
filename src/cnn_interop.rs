use crate::data::{load_image_dataset, Dataset3D};
use crate::model::{Hyperparameters, CNN};
use crate::optimizer::{str_to_optmizer, Optimizer};
use crate::util::c_str_to_rust_str;
use libc::{c_char, c_double, c_longlong, size_t};
use ndarray::Array3;
use std::ffi::CString;

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

pub unsafe extern "C" fn fit_cnn(cnn_ptr: *mut CNN) {
    unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.").fit() }
}

pub unsafe extern "C" fn predict_cnn(
    cnn_ptr: *mut CNN,
    input: *const *const *const c_double,
    nrows: size_t,
    ncols: size_t,
    nchannels: size_t,
) -> *const c_double {
    // Convert input to Array3
    let input: Array3<f64> = unsafe {
        Array3::from_shape_fn((nrows, ncols, nchannels), |(i, j, k)| {
            *(*(*input.add(i)).add(j)).add(k)
        })
    };

    let cnn = unsafe { cnn_ptr.as_mut().expect("Null CNN pointer.") };
    let output = cnn.predict(input);
    let output_ptr = output.as_ptr();
    std::mem::forget(output);

    output_ptr
}

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

    full_path.as_ptr()
}

pub unsafe extern "C" fn load_cnn(model_path: *const c_char) -> *mut CNN {
    Box::leak(Box::new(CNN::load(c_str_to_rust_str(model_path))))
}

pub unsafe extern "C" fn free_cnn(cnn_ptr: *mut CNN) {
    let _ = unsafe { Box::from_raw(cnn_ptr) };
}
