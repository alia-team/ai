use std::ffi::CString;
use std::slice;
use ai::data_processing;
use std::ffi::CStr;


#[test]
fn image_to_vector() {
    let img_path = CString::new("tests/test_images/5x5-rgb-square.png").unwrap();

    let pixel_values = unsafe {
        let pixel_values_ptr = data_processing::image_to_vector(img_path.as_ptr());
        let expected_values = vec![
            255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0,
            0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0,
            0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
            0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0,
            0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
        ];

        let pixel_values_slice = slice::from_raw_parts(pixel_values_ptr, expected_values.len());
        assert_eq!(pixel_values_slice, expected_values.as_slice());
    };
}

#[test]
fn get_all_images_in_folder() {
    let folder_path = CString::new("tests\\test_images").unwrap();
    let images_ptr = unsafe {
        data_processing::get_all_images_in_folder(folder_path.as_ptr())
    };

    let images_len = unsafe {
        let images_slice = slice::from_raw_parts(images_ptr, 541);
        images_slice.len()
    };

    assert_eq!(images_len, 541);
}
