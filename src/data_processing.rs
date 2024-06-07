use image::{GenericImageView, Pixel};

#[derive(Debug, PartialEq)]
pub enum ImageError {
    ImageNotProcceded,
}

pub fn image_to_vector(image_path: &str) -> Result<Vec<f64>, ImageError> {
    let img = image::open(image_path).expect("Failed to open image");

    let (width, height) = img.dimensions();

    let mut pixel_values = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let channels = pixel.to_rgb().0;
            pixel_values.extend(channels.iter().map(|&v| v as f64));
        }
    }
    Ok(pixel_values)
}

pub fn get_all_images_in_folder(folder_path: &str) -> Result<Vec<Vec<f64>>, ImageError> {
    let paths = std::fs::read_dir(folder_path).expect("Failed to read directory");

    let mut images = Vec::new();

    for path in paths {
        let path = path.expect("Failed to get path").path();
        let path_str = path.to_str().expect("Failed to convert path to string");

        let image = image_to_vector(path_str)?;
        images.push(image);
    }

    Ok(images)
}
