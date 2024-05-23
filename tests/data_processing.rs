use ai::data_processing;

#[test]
fn image_to_vector() {
    let img_path = "tests/test_images/5x5-rgb-square.png";

    let expected_values = vec![
        255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0,
        0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0,
        0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
        0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0,
        0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0,
    ];

    let pixel_values = data_processing::image_to_vector(img_path).unwrap();

    assert_eq!(
        pixel_values[..expected_values.len()],
        expected_values[..],
        "Pixel values do not match the expected values"
    );
}

