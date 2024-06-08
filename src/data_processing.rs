use image::{GenericImageView, Pixel};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::str;

#[no_mangle]
pub extern "C" fn image_to_vector(image_path: *const c_char) -> *mut f64 {
    let c_str = unsafe { CStr::from_ptr(image_path) };
    let mut path = String::from(c_str.to_str().unwrap());

    let img = match image::open(&path) {
        Ok(img) => img,
        Err(err) => {
            println!("Error while opening image {}: {}", path, err);
            path.clear();
            return std::ptr::null_mut();
        }
    };

    let (width, height) = img.dimensions();

    let mut pixel_values = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let channels = pixel.to_rgb().0;
            pixel_values.extend(channels.iter().map(|&v| v as f64));
        }
    }

    path.clear();

    let pixel_values_raw = pixel_values.into_boxed_slice();
    Box::into_raw(pixel_values_raw) as *mut f64
}

#[no_mangle]
pub extern "C" fn get_all_images_in_folder(folder_path: *const c_char) -> *mut HashMap<*mut f64, *mut c_char> {
    let c_str = unsafe { CStr::from_ptr(folder_path) };
    let mut path = String::from(c_str.to_str().unwrap());

    let mut images_and_labels = HashMap::new();

    for entry in std::fs::read_dir(&path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            let label = CString::new(path.file_name().unwrap().to_str().unwrap()).unwrap();
            let label_ptr = label.into_raw();

            for image_path in std::fs::read_dir(path).unwrap() {
                let image_path = image_path.unwrap().path();
                let image_ptr = image_to_vector(image_path.to_str().unwrap().as_ptr() as *const c_char);
                if !image_ptr.is_null() {
                    images_and_labels.insert(image_ptr, label_ptr);
                }
            }
        }
    }

    path.clear();

    let images_and_labels_raw = Box::new(images_and_labels);
    Box::into_raw(images_and_labels_raw) as *mut HashMap<*mut f64, *mut c_char>
}
