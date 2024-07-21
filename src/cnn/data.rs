use image::io::Reader as ImageReader;
use ndarray::Array3;
use rand::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct TrainingData {
    trn_img: Vec<Array3<f32>>,
    trn_lbl: Vec<usize>,
    tst_img: Vec<Array3<f32>>,
    tst_lbl: Vec<usize>,
    pub trn_size: usize,
    pub tst_size: usize,
    pub classes: HashMap<usize, usize>,
}

impl TrainingData {
    pub fn get_random_image(&self) -> Result<(Array3<f32>, usize), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.trn_size);
        let img = self.trn_img[index].to_owned();
        Ok((img, self.trn_lbl[index]))
    }

    pub fn get_random_test_image(&self) -> Result<(Array3<f32>, usize), String> {
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.tst_size);
        let img = self.tst_img[index].to_owned();
        Ok((img, self.tst_lbl[index]))
    }
}

pub fn load_image_dataset<T>(
    dataset_path: T,
    train_ratio: f32,
    max_images_per_class: Option<usize>,
) -> Result<TrainingData, String>
where
    T: AsRef<Path>,
{
    let dataset_path = dataset_path.as_ref();
    let mut trn_img = Vec::new();
    let mut trn_lbl = Vec::new();
    let mut tst_img = Vec::new();
    let mut tst_lbl = Vec::new();
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
        let num_train = (num_images as f32 * train_ratio).round() as usize;

        for (i, img_path) in images.into_iter().take(num_images).enumerate() {
            if i < num_train {
                trn_img.push(load_image(&img_path).expect("Training image not found."));
                trn_lbl.push(class_index);
            } else {
                tst_img.push(load_image(&img_path).expect("Test image not found."));
                tst_lbl.push(class_index);
            }
        }
    }

    let trn_size = trn_img.len();
    let tst_size = tst_img.len();

    Ok(TrainingData {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        trn_size,
        tst_size,
        classes,
    })
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
        array[[y as usize, x as usize, 0]] = r / 255.0;
        array[[y as usize, x as usize, 1]] = g / 255.0;
        array[[y as usize, x as usize, 2]] = b / 255.0;
    }
    Ok(array)
}
