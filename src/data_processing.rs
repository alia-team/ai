use image::{GenericImageView, Pixel};
use std::ffi::CStr;
use std::str;

#[no_mangle]
pub extern "C" fn image_to_vector(image_path: *const i8) -> *mut f64 {
    let c_str = unsafe { std::ffi::CStr::from_ptr(image_path) };
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
pub extern "C" fn get_all_images_in_folder(folder_path: *const i8) -> *mut *mut f64 {
    let c_str = unsafe { CStr::from_ptr(folder_path) };
    let mut path = String::from(str::from_utf8(c_str.to_bytes()).unwrap());

    let paths = match std::fs::read_dir(&path) {
        Ok(paths) => paths,
        Err(err) => {
            println!("Error while reading directory {}: {}", path, err);
            path.clear();
            return std::ptr::null_mut();
        }
    };

    let mut images = Vec::new();

    for path in paths {
        let path = match path {
            Ok(path) => path.path(),
            Err(err) => {
                println!("Error while reading path: {}", err);
                continue;
            }
        };
        let mut path_str = path.to_str().unwrap().to_owned();

        let image_ptr = image_to_vector(path_str.as_ptr() as *const i8);
        if !image_ptr.is_null() {
            images.push(image_ptr);
        }

        path_str.clear();
    }

    path.clear();

    let images_raw = images.into_boxed_slice();
    Box::into_raw(images_raw) as *mut *mut f64
}
