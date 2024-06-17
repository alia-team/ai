# AI

Our from scratch AI library for building neural networks.

## Contributing

Before working on this project, you must read and follow the
[guidelines](https://github.com/alia-team/ai/blob/main/CONTRIBUTING.md).

## Modules

There are modules for each type of neural network.
There are also helpers modules for data processing and utilities.

### Activation

Contains all activation functions.

### Data Processing

Contains functions for converting images to vectors and for building/processing
datasets.

### Linear Regression

Contains a simple linear model for regression tasks.

### MLP

Contains a Multi-Layer Perceptron structure with all its methods.

### Naive RBF

Contains the structure for an RBF neural network, where there are **as many**
**centroids as the number of samples**in the training dataset.
This method is a simpler way to implement an RBF for tasks where we exactly
know all the possible inputs, because it's an obvious case of overfitting (on
purpose).
The only hyperparameter is the**gamma**.

### Perceptron

Contains a simple perceptron structure with all its methods.

### RBF

Contains the structure for an RBF neural network, where the **centroids are**
**picked up randomly from the training dataset**.
It uses **K-means clustering** with the **Lloyd algorithm**.
The hyperparameters are the **gamma**, the **number of centroids** and the
**maximum number of iterations** for the Lloyd algorithm.

### Utils

Contains initialization functions (for weights, layers outputs, ...), some maths
(euclidian distance), interoperability functions, and others.

## Commands

### Building

To make the project interoperable, it must be built using the
`--release` flag:

```console
cargo build --release
```

### Testing

- For running all tests:

```console
cargo test
```

- For running all tests and show prints (with `println!()` for example):

```console
cargo test -- --nocapture
```

- For running a specific test:

```console
cargo test test_name -- --exact
```
