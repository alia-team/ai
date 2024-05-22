use image::{GenericImageView, Pixel};

#[derive(Debug, PartialEq)]
pub enum ImageError {
    ImageNotProcceded,
}

pub fn image_to_array(image_path:&str) ->Result<Vec<u8>, ImageError>{
    let img = image::open(image_path).expect("Failed to open image");

    let (width, height) = img.dimensions();

    let mut pixel_values = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let channels = pixel.to_rgb().0;
            pixel_values.extend_from_slice(&channels);
        }
    }
    Ok(pixel_values)
}