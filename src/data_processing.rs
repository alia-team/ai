use image::{GenericImageView, Pixel};
use std::ffi::CStr;
use std::os::raw::c_char;

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

    let mut pixel_values = Vec::with_capacity((width * height * 1) as usize);

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
