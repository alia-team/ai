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

fn train(
    cnn: &mut CNN,
    inputs: &[Array3<f32>],
    targets: &[usize],
    epochs: usize,
    batch_size: usize,
    initial_lr: f32,
) {
    let mut adam = Adam::new(initial_lr, 0.9, 0.999, 1e-8);
    let lr_scheduler = LRScheduler::new(initial_lr, 0.95, 1000);
    let num_batches = inputs.len() / batch_size;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for batch in 0..num_batches {
            println!("Batch {}/{}...", batch + 1, num_batches);
            let start = batch * batch_size;
            let end = start + batch_size;
            let batch_inputs = &inputs[start..end];
            let batch_targets = &targets[start..end];

            let (batch_loss, batch_grads) = process_batch(cnn, batch_inputs, batch_targets);
            println!("Batch loss: {}", batch_loss);

            total_loss += batch_loss;

            // Update learning rate
            adam.lr = lr_scheduler.get_lr(epoch * num_batches + batch);

            // Update parameters using Adam
            adam.update(
                &mut [
                    &mut cnn.conv1.kernel.view_mut().into_dyn(),
                    &mut cnn.conv1.bias.view_mut().into_dyn(),
                    &mut cnn.bn1.gamma.view_mut().into_dyn(),
                    &mut cnn.bn1.beta.view_mut().into_dyn(),
                    &mut cnn.conv2.kernel.view_mut().into_dyn(),
                    &mut cnn.conv2.bias.view_mut().into_dyn(),
                    &mut cnn.bn2.gamma.view_mut().into_dyn(),
                    &mut cnn.bn2.beta.view_mut().into_dyn(),
                    &mut cnn.conv3.kernel.view_mut().into_dyn(),
                    &mut cnn.conv3.bias.view_mut().into_dyn(),
                    &mut cnn.bn3.gamma.view_mut().into_dyn(),
                    &mut cnn.bn3.beta.view_mut().into_dyn(),
                    &mut cnn.dense1.weights.view_mut().into_dyn(),
                    &mut cnn.dense1.bias.view_mut().into_dyn(),
                    &mut cnn.bn4.gamma.view_mut().into_dyn(),
                    &mut cnn.bn4.beta.view_mut().into_dyn(),
                    &mut cnn.dense2.weights.view_mut().into_dyn(),
                    &mut cnn.dense2.bias.view_mut().into_dyn(),
                ],
                &batch_grads.iter().map(|g| g.view()).collect::<Vec<_>>(),
            );
        }

        println!(
            "Epoch {}/{}: Loss = {}, LR = {}",
            epoch + 1,
            epochs,
            total_loss / (num_batches as f32),
            adam.lr
        );
    }
}

fn process_batch(
    cnn: &mut CNN,
    inputs: &[Array3<f32>],
    targets: &[usize],
) -> (f32, Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>) {
    let batch_size = inputs.len();
    let mut sample_counter: usize = 0;
    let mut total_loss = 0.0;
    let mut total_grads: Option<Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>> = None;

    for (input, &target) in inputs.iter().zip(targets.iter()) {
        match cnn.backward(input, target) {
            Ok((sample_loss, sample_grads)) => {
                sample_counter += 1;
                println!(
                    "Sample loss {}/{}: {}",
                    sample_counter, batch_size, sample_loss
                );
                total_loss += sample_loss;

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

    (batch_loss, batch_grads)
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
    train(&mut cnn, &samples, &targets, 1, 32, 0.001);
    cnn.save_weights("weights.json").unwrap();

    CNN::load_weights("weights.json").unwrap();
}
