use rand::Rng;

pub struct Perceptron {
    weights: Vec<f64>, // Dynamic number of weights, including the bias
}

impl Perceptron {
    // Initialize the Perceptron with random weights
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..=input_size)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
            .collect();
        Perceptron { weights }
    }

    // Train the Perceptron with the given data points and labels
    pub fn fit(&mut self, data_points: &[Vec<f64>], class_labels: &[f64], iterations: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..iterations {
            let random_index = rng.gen_range(0..data_points.len());
            let target_label = class_labels[random_index];
            let mut input = vec![1.0]; // Bias term
            input.extend(&data_points[random_index]);

            // Calculate weighted sum
            let weighted_sum: f64 = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();

            // Activation function
            let predicted_label = if weighted_sum >= 0.0 { 1.0 } else { -1.0 };

            // Update weights
            (0..self.weights.len()).for_each(|i| {
                self.weights[i] += 0.001 * (target_label - predicted_label) * input[i];
            });
        }
    }

    // Predict the class of a new data point
    pub fn predict(&self, new_point: &[f64]) -> f64 {
        let mut input = vec![1.0]; // Bias term
        input.extend(new_point);
        let weighted_sum: f64 = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();
        if weighted_sum >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let mut rng = rand::thread_rng();
        let data_points: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0])
            .collect();

        let mut class_labels: Vec<f64> = Vec::new();
        for point in &data_points {
            if point[0] + 3.0 * point[1] - 78.0 * 3.0 >= 0.0 {
                class_labels.push(1.0);
            } else {
                class_labels.push(-1.0);
            }
        }

        let mut perceptron = Perceptron::new(2);
        perceptron.fit(&data_points, &class_labels, 1000);

        let new_point = vec![0.0, 0.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![50.0, 50.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![0.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![100.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);
    }
}
