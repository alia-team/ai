use ai::data_processing;
use std::ffi::CString;
use std::slice;

#[test]
fn image_to_vector() {
    let img_path = CString::new("tests/test_images/5x5-rgb-square.png").unwrap();

    let _ = unsafe {
        let pixel_values_ptr = data_processing::image_to_vector(img_path.as_ptr());
        let expected_values = vec![
            255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0,
            0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
            0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0,
            255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0,
            0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
        ];

        let pixel_values_slice = slice::from_raw_parts(pixel_values_ptr, expected_values.len());
        assert_eq!(pixel_values_slice, expected_values.as_slice());
    };
}
