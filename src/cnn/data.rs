use image::io::Reader as ImageReader;
use ndarray::Array3;
use rand::prelude::IteratorRandom;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// On-demand loading of images and full dataset loading
#[derive(Clone)]
pub enum TrainImage {
    Image(Array3<f32>),
    Path(PathBuf),
}

#[derive(Default)]
pub struct TrainingData {
    pub trn_img: Vec<TrainImage>,
    pub trn_lbl: Vec<usize>,
    pub tst_img: Vec<TrainImage>,
    pub tst_lbl: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub trn_size: usize,
    pub tst_size: usize,
    pub classes: HashMap<usize, usize>,
}

pub fn load_image(path: &Path) -> Result<Array3<f32>, String> {
    let img = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let img = img.to_rgb8();

    let rows = img.height() as usize;
    let cols = img.width() as usize;
    let mut array = Array3::zeros((rows, cols, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        array[[x as usize, y as usize, 0]] = r / 255.0;
        array[[x as usize, y as usize, 1]] = g / 255.0;
        array[[x as usize, y as usize, 2]] = b / 255.0;
    }

    Ok(array)
}

pub fn get_random_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data
        .trn_img
        .iter()
        .zip(data.trn_lbl.iter())
        .choose(&mut rng)
        .unwrap();
    match img {
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_image(img_path).unwrap();
            (img, *label)
        }
    }
}

pub fn get_random_test_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data
        .tst_img
        .iter()
        .zip(data.tst_lbl.iter())
        .choose(&mut rng)
        .unwrap();
    match img {
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_image(img_path).unwrap();
            (img, *label)
        }
    }
}
