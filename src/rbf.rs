extern crate rand;
use rand::Rng;

#[derive(Debug, PartialEq)]
pub struct Centroid {
    pub coordinates: Vec<f64>,
}

impl Centroid {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Centroid { coordinates }
    }

    pub fn forward(&self, input: Vec<f64>, gamma: f64) -> f64 {
        let mut vec_sub: Vec<f64> = vec![];
        for (i, value) in input.iter().enumerate() {
            vec_sub.push(value - self.coordinates[i])
        }

        let mut norm: f64 = 0.0;
        for value in vec_sub {
            norm += value.powi(2)
        }
        norm = norm.sqrt();

        (-gamma * norm.powi(2)).exp()
    }
}

pub struct RBF {
    pub neurons_per_layer: Vec<usize>,
    pub centroids: Vec<Centroid>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub is_classification: bool,
}

impl RBF {
    pub fn initialize_centroids(&mut self, data: &Vec<Vec<f64>>, k: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..k {
            let index = rng.gen_range(0..data.len());
            self.centroids.push(Centroid::new(data[index].clone()));
        }
    }

    pub fn lloyds_algorithm(&mut self, data: &Vec<Vec<f64>>, max_iterations: usize) {
        let k = self.centroids.len();
        for _ in 0..max_iterations {
            // Assignment step
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![vec![]; k];
            for point in data {
                let mut min_dist = f64::MAX;
                let mut min_index = 0;
                for (i, centroid) in self.centroids.iter().enumerate() {
                    let dist = euclidean_distance(point, &centroid.coordinates);
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                clusters[min_index].push(point.clone());
            }

            // Update step
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = compute_centroid(cluster);
                    self.centroids[i] = Centroid::new(new_centroid);
                }
            }
        }
    }
}

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn compute_centroid(cluster: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut centroid = vec![0.0; cluster[0].len()];
    for point in cluster {
        for (i, value) in point.iter().enumerate() {
            centroid[i] += value;
        }
    }
    for value in centroid.iter_mut() {
        *value /= cluster.len() as f64;
    }
    centroid
}
