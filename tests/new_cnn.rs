use ai::fit::sparse_categorical_crossentropy;
use ai::{
    cnn::CNN,
    fit::{Adam, LRScheduler},
};
use env_logger::Env;
use ndarray::{Array1, Array3, ArrayBase, ArrayViewMut1, IxDyn, OwnedRepr};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|a| (a - max_val).exp());
    let sum = exp.sum();
    exp / sum
}

fn calculate_accuracy(predictions: &[Array1<f32>], targets: &[usize]) -> f32 {
    let correct = predictions
        .iter()
        .zip(targets)
        .filter(|(pred, &target)| {
            pred.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
                == target
        })
        .count();

    correct as f32 / predictions.len() as f32
}

fn l2_regularization(cnn: &CNN, lambda: f32) -> f32 {
    let mut reg_loss = 0.0;
    reg_loss += cnn.conv1.kernel.mapv(|x| x * x).sum();
    reg_loss += cnn.dense1.weights.mapv(|x| x * x).sum();
    reg_loss += cnn.dense2.weights.mapv(|x| x * x).sum();
    0.5 * lambda * reg_loss
}

fn train(
    cnn: &mut CNN,
    inputs: &[Array3<f32>],
    targets: &[usize],
    epochs: usize,
    batch_size: usize,
    initial_lr: f32,
    lambda: f32,
) {
    let mut adam = Adam::new(initial_lr, 0.9, 0.999, 1e-8);
    let lr_scheduler = LRScheduler::new(initial_lr, 0.95, 1000);
    let num_batches = inputs.len() / batch_size;

    let mut model_loss = 0.0;
    let mut total_correct = 0;

    for epoch in 0..epochs {
        for batch in 0..num_batches {
            println!("Batch {}/{}...", batch + 1, num_batches);
            let start = batch * batch_size;
            let end = start + batch_size;
            let batch_inputs = &inputs[start..end];
            let batch_targets = &targets[start..end];

            let (batch_loss, batch_grads, batch_correct, batch_samples) =
                process_batch(cnn, batch_inputs, batch_targets, lambda);

            let batch_accuracy = batch_correct as f32 / batch_samples as f32 * 100.;

            model_loss += batch_loss;
            total_correct += batch_correct;

            println!();
            println!(
                "Batch loss: {}, Batch accuracy: {}%",
                batch_loss, batch_accuracy
            );

            // Update learning rate
            adam.lr = lr_scheduler.get_lr(epoch * num_batches + batch);

            // Update parameters using Adam
            adam.update(
                &mut [
                    &mut cnn.conv1.kernel.view_mut().into_dyn(),
                    &mut cnn.conv1.bias.view_mut().into_dyn(),
                    &mut cnn.dense1.weights.view_mut().into_dyn(),
                    &mut cnn.dense1.bias.view_mut().into_dyn(),
                    &mut cnn.dense2.weights.view_mut().into_dyn(),
                    &mut cnn.dense2.bias.view_mut().into_dyn(),
                ],
                &batch_grads.iter().map(|g| g.view()).collect::<Vec<_>>(),
            );
        }
    }

    println!();
    println!("Model loss: {}", model_loss);
    println!(
        "Model accuracy: {}%",
        total_correct as f32 / (batch_size * num_batches * epochs) as f32 * 100.
    );
}

fn process_batch(
    cnn: &mut CNN,
    inputs: &[Array3<f32>],
    targets: &[usize],
    lambda: f32,
) -> (f32, Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>, usize, usize) {
    let batch_size = inputs.len();
    let mut sample_counter: usize = 0;
    let mut total_loss = 0.0;
    let mut total_grads: Option<Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>> = None;
    let mut correct_predictions = 0;

    for (input, &target) in inputs.iter().zip(targets.iter()) {
        sample_counter += 1;
        println!();
        println!("Sample {}/{}", sample_counter, batch_size);
        let prediction = cnn.forward(input);
        let sample_loss = sparse_categorical_crossentropy(target, &prediction, false);

        match cnn.backward(input, target) {
            Ok((_, sample_grads)) => {
                // Add L2 regularization to the loss
                let l2_loss = l2_regularization(cnn, lambda);
                let total_sample_loss = sample_loss + l2_loss;

                println!(
                    "Loss {} (CE: {}, L2: {})",
                    total_sample_loss, sample_loss, l2_loss
                );
                total_loss += total_sample_loss;

                // Calculate accuracy
                let predicted_class = prediction
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap();

                if predicted_class == target {
                    correct_predictions += 1;
                }

                println!(
                    "Batch accuracy: {}%",
                    correct_predictions as f32 / sample_counter as f32 * 100.
                );

                // Accumulate gradients
                total_grads = Some(match total_grads {
                    Some(grads) => grads
                        .iter()
                        .zip(&sample_grads)
                        .map(|(a, b)| {
                            let mut result = a.clone();
                            result += &b.view();
                            result
                        })
                        .collect(),
                    None => sample_grads,
                });
            }
            Err(e) => {
                println!("Error processing sample: {}", e);
                continue;
            }
        }
    }

    let batch_loss = if sample_counter > 0 {
        total_loss / sample_counter as f32
    } else {
        0.0
    };

    let batch_grads = total_grads
        .unwrap_or_default()
        .into_iter()
        .map(|g| g.mapv(|v| v / sample_counter.max(1) as f32))
        .collect();

    (batch_loss, batch_grads, correct_predictions, sample_counter)
}

fn load_mnist(file_path: &str) -> (Vec<Array3<f32>>, Vec<usize>) {
    println!("Loading MNIST...");
    let file = File::open(file_path).expect("Failed to open MNIST dataset file");
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    let mut targets = Vec::new();

    let mut counter: usize = 0;

    for line in reader.lines().skip(1) {
        // Skip header row if present
        let line = line.expect("Failed to read line");
        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.parse().expect("Failed to parse value"))
            .collect();

        let label = values[0] as usize;
        let pixels: Vec<f32> = values[1..]
            .iter()
            .map(|&x| (x / 255.0 - 0.5) / 0.5)
            .collect();
        let image = Array3::from_shape_vec((28, 28, 1), pixels)
            .expect("Failed to create Array3 from pixel values");

        samples.push(image);
        targets.push(label);

        counter += 1;
    }
    println!("Loaded {} samples with their label.", counter);

    (samples, targets)
}

#[test]
fn cnn() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

    let input_shape = (28, 28, 1); // Example input shape
    let mut cnn = CNN::new(input_shape);

    // Prepare your training data
    let (samples, targets) = load_mnist("train.csv");

    // Train the model
    train(&mut cnn, &samples, &targets, 1, 32, 0.0001, 0.0001);
    cnn.save_weights("weights.json").unwrap();
}
