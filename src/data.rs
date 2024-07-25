use image::io::Reader as ImageReader;
use itertools::Itertools;
use ndarray::{Array1, Array3};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub trait Dataset {
    type Array;
    fn get_random_training_sample(&self) -> Result<(Self::Array, u8), String>;
    fn get_random_testing_sample(&self) -> Result<(Self::Array, u8), String>;
}

#[derive(Serialize, Deserialize)]
pub struct Dataset1D {
    training_samples: Vec<Array1<f64>>,
    training_targets: Vec<u8>,
    testing_samples: Vec<Array1<f64>>,
    testing_targets: Vec<u8>,
    pub training_size: usize,
    pub testing_size: usize,
    pub classes: HashMap<u8, u8>,
}

impl Dataset1D {
    pub fn new(
        samples: Vec<Array1<f64>>,
        targets: Vec<u8>,
        train_ratio: f64,
        max_samples_per_class: Option<u8>,
    ) -> Self {
        let mut training_samples: Vec<Array1<f64>> = Vec::new();
        let mut training_targets: Vec<u8> = Vec::new();
        let mut testing_samples: Vec<Array1<f64>> = Vec::new();
        let mut testing_targets: Vec<u8> = Vec::new();
        let mut classes: HashMap<u8, u8> = HashMap::new();

        let num_samples = max_samples_per_class.unwrap_or(samples.len() as u8);
        let num_training_samples = (num_samples as f64 * train_ratio).round() as usize;

        for (i, sample) in samples.into_iter().take(num_samples as usize).enumerate() {
            if i < num_training_samples {
                training_samples.push(sample);
                training_targets.push(targets[i]);
            } else {
                testing_samples.push(sample);
                testing_targets.push(targets[i]);
            }
        }

        let training_size: usize = training_samples.len();
        let testing_size: usize = testing_samples.len();

        // Build classes hashmap
        let mut sorted_unique_targets: Vec<u8> = targets.clone();
        sorted_unique_targets.sort();
        sorted_unique_targets = sorted_unique_targets.into_iter().unique().collect();
        for target in sorted_unique_targets {
            classes.insert(target, target);
        }

        Dataset1D {
            training_samples,
            training_targets,
            testing_samples,
            testing_targets,
            training_size,
            testing_size,
            classes,
        }
    }
}

impl Dataset for Dataset1D {
    type Array = Array1<f64>;

    fn get_random_training_sample(&self) -> Result<(Self::Array, u8), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.training_size);
        Ok((
            self.training_samples[index].to_owned(),
            self.training_targets[index],
        ))
    }

    fn get_random_testing_sample(&self) -> Result<(Self::Array, u8), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.testing_size);
        Ok((
            self.testing_samples[index].to_owned(),
            self.testing_targets[index],
        ))
    }
}

#[derive(Deserialize, Serialize)]
pub struct Dataset3D {
    training_samples: Vec<Array3<f64>>,
    training_targets: Vec<usize>,
    testing_samples: Vec<Array3<f64>>,
    testing_targets: Vec<usize>,
    pub training_size: usize,
    pub testing_size: usize,
    pub classes: HashMap<usize, usize>,
}

impl Dataset3D {
    pub fn get_random_sample(&self) -> Result<(Array3<f64>, usize), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.training_size);
        Ok((
            self.training_samples[index].to_owned(),
            self.training_targets[index],
        ))
    }

    pub fn get_random_test_sample(&self) -> Result<(Array3<f64>, usize), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.testing_size);
        Ok((
            self.testing_samples[index].to_owned(),
            self.testing_targets[index],
        ))
    }
}

pub fn load_image_dataset<T>(
    dataset_path: T,
    train_ratio: f64,
    max_images_per_class: Option<usize>,
) -> Result<Dataset3D, String>
where
    T: AsRef<Path>,
{
    let dataset_path = dataset_path.as_ref();
    let mut training_samples = Vec::new();
    let mut training_targets = Vec::new();
    let mut testing_samples = Vec::new();
    let mut testing_targets = Vec::new();
    let mut classes = HashMap::new();

    let mut class_dirs: Vec<_> = fs::read_dir(dataset_path)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_dir() {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // Sort class directories alphabetically
    class_dirs.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());

    for (class_index, class_path) in class_dirs.into_iter().enumerate() {
        classes.insert(class_index, class_index);

        let images: Vec<_> = fs::read_dir(&class_path)
            .map_err(|e| e.to_string())?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path
                    .extension()
                    .map_or(false, |ext| ext == "png" || ext == "jpg" || ext == "jpeg")
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        let num_images = max_images_per_class.unwrap_or(images.len());
        let num_train = (num_images as f64 * train_ratio).round() as usize;

        for (i, img_path) in images.into_iter().take(num_images).enumerate() {
            if i < num_train {
                training_samples.push(load_image(&img_path).expect("Training image not found."));
                training_targets.push(class_index);
            } else {
                testing_samples.push(load_image(&img_path).expect("Test image not found."));
                testing_targets.push(class_index);
            }
        }
    }

    let training_size = training_samples.len();
    let testing_size = testing_samples.len();

    println!(
        "Loaded {} training images and {} testing images.",
        training_size, testing_size
    );

    Ok(Dataset3D {
        training_samples,
        training_targets,
        testing_samples,
        testing_targets,
        training_size,
        testing_size,
        classes,
    })
}

pub fn load_image(path: &Path) -> Result<Array3<f64>, String> {
    let img = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let img = img.to_rgb8();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    let mut array = Array3::zeros((rows, cols, 3));
    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f64, pixel[1] as f64, pixel[2] as f64);
        array[[y as usize, x as usize, 0]] = r / 255.0;
        array[[y as usize, x as usize, 1]] = g / 255.0;
        array[[y as usize, x as usize, 2]] = b / 255.0;
    }
    Ok(array)
}
