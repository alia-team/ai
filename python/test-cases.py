from mlp import MLP

# Example usage:
npl = (2, 1)
mlp = MLP(npl)

training_dataset = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
]
labels = [
    [-1.0],
    [1.0],
    [1.0]
]

mlp.train(training_dataset, labels, 0.1, 1000000, True)

for k in range(len(training_dataset)):
    output = mlp.predict(training_dataset[k], True)
    print(output)
