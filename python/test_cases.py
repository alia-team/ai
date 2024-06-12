from mlp import MLP
import matplotlib.pyplot as plt
import numpy as np

# Linear Simple
# MLP (2, 1)   : OK
npl = (2, 1)
mlp = MLP(npl)

training_dataset = np.array([
      [1, 1],
      [2, 3],
      [3, 3]
])
labels = np.array([
    [1.0],
    [-1.0],
    [-1.0]
])

mlp.train(training_dataset, labels, 0.1, 1000000, True)

new_point = [1, 1]
result = mlp.predict(new_point, True)
print('Linear simple:', result)


# Linear Multiple :
# MLP (2, 1): OK
npl = (2, 1)
mlp = MLP(npl)

training_dataset = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
labels = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

mlp.train(training_dataset, labels, 0.1, 10000, True)

new_point = [2.75, 2.75]
result = mlp.predict(new_point, True)
print('Linear Multiple:', result)

# XOR
# MLP (2, 2, 1): 
npl = (2, 4, 1)
mlp = MLP(npl)
training_dataset = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
labels = np.array([[1], [1], [-1], [-1]])

mlp.train(training_dataset, labels, 0.1, 1000000, True)

new_point = [1, 1]
result = mlp.predict(new_point, True)
print('XOR:', result)


# Cross
# MLP (2, 4, 1)
npl = (2, 5, 1)
mlp = MLP(npl)

# Initialize the training dataset and labels
training_dataset = np.random.random((500, 2)) * 2.0 - 1.0
labels = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in training_dataset])
labels = labels.reshape(-1, 1)  # Ensure labels are in the correct shape

mlp.train(training_dataset, labels, 0.1, 1000000, True)

new_point = [-1, -1]
result = mlp.predict(new_point, True)
print('Cross:', result)


# Multi linear 3 classes
# MLP (2, 3)
npl = (2, 3)
mlp = MLP(npl)

# Initialize the training dataset and labels
training_dataset = np.random.random((500, 2)) * 2.0 - 1.0
labels = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [-1, -1, -1]for p in training_dataset])

training_dataset = training_dataset[[not np.all(arr == [0, 0, 0]) for arr in labels]]
labels = labels[[not np.all(arr == [0, 0, 0]) for arr in labels]]

mlp.train(training_dataset, labels, 0.1, 1000, True)

new_point = [1, 1]
result = mlp.predict(new_point, True)
print('Multi linear 3 classes:', result)